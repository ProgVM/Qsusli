import telebot
import json
import os
import threading
import time
import logging
import torch
import random
import traceback
from qsusli_model import GlobalWorkspace, load_model, generate_text, tokens_to_indices, split_text_to_sentences, fine_tune, search_wikipedia, generate_image_from_text, process_rlhf_correction
from db import DB

db = DB()

API_TOKEN = '8034310036:AAFLGX0lEtnOdFAj7OSurkNz7e3dVCorE2U'

bot = telebot.TeleBot(API_TOKEN)

model_path = 'qsusli_model.pth'

workspaces = {}
activity_settings = {}
user_last_outputs = {}
user_training_buffers = {}

if os.path.exists(model_path):
    print("Загрузка модели...")
    model, optimizer, bpe = load_model(model_path)
else:
    print("Файл модели не найден. Обучение новой модели...")
    from qsusli_model import train_new_model
    model, optimizer, bpe = train_new_model()

inv_vocab = {v: k for k, v in bpe.vocab.items()}
is_finetuning = False
train_data_lock = threading.Lock()
fine_tune_threshold = 10

# Настройка логирования
logging.basicConfig(filename='bot.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def load_activity_settings():
    global activity_settings
    raw_settings = db.load_all_group_settings()
    activity_settings.clear()
    for chat_id, settings in raw_settings.items():
        level_str = settings.get('activity_level')
        if level_str is not None:
            try:
                activity_settings[chat_id] = int(level_str)
            except ValueError:
                pass
    return activity_settings

load_activity_settings()

def save_activity_settings():
    for chat_id, level in activity_settings.items():
        db.set_group_setting(chat_id, 'activity_level', level)

# Загружаем настройки активности при старте
load_activity_settings()

# Система ограничения частоты запросов
last_message_time = {}
message_count = {}
RATE_LIMIT_SECONDS = 1.5  # Минимальная задержка между сообщениями
MAX_MESSAGES_PER_MINUTE = 20  # Максимум сообщений в минуту

def can_send_message(chat_id):
    """Проверяет можно ли отправить сообщение (rate limiting)"""
    current_time = time.time()
    
    # Проверяем последнее время отправки
    if chat_id in last_message_time:
        time_diff = current_time - last_message_time[chat_id]
        if time_diff < RATE_LIMIT_SECONDS:
            return False
    
    # Проверяем количество сообщений в минуту
    if chat_id in message_count:
        message_count[chat_id] = [(t, c) for t, c in message_count[chat_id] if current_time - t < 60]
        if len(message_count[chat_id]) >= MAX_MESSAGES_PER_MINUTE:
            return False
    else:
        message_count[chat_id] = []
    
    return True

def safe_send_message(message, text):
    """Безопасная отправка сообщений с обработкой ошибок"""
    try:
        chat_id = message.chat.id
        
        # Проверяем лимиты
        if not can_send_message(chat_id):
            logging.info(f"Rate limit: пропуск сообщения для чата {chat_id}")
            return False
        
        # Отправляем сообщение
        bot.reply_to(message, text)
        
        # Обновляем счетчики
        current_time = time.time()
        last_message_time[chat_id] = current_time
        if chat_id not in message_count:
            message_count[chat_id] = []
        message_count[chat_id].append((current_time, 1))
        
        return True
        
    except telebot.apihelper.ApiTelegramException as e:
        if e.error_code == 429:  # Too Many Requests
            retry_after = 60  # По умолчанию 60 секунд
            try:
                # Пытаемся извлечь время ожидания из описания
                if "retry after" in e.description:
                    retry_after = int(e.description.split("retry after")[1].strip())
            except:
                pass
            
            logging.warning(f"Rate limit hit for chat {chat_id}, waiting {retry_after} seconds")
            time.sleep(min(retry_after, 120))  # Максимум 2 минуты ожидания
            return False
            
        else:
            logging.error(f"Telegram API error: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return False

def load_train_data():
    return db.load_train_data()


def save_train_data(train_pairs):
    # train_pairs — список кортежей (input_text, output_text)
    for input_text, output_text in train_pairs:
        db.save_train_pair(input_text, output_text)


def clear_train_data():
    db.clear_train_data()


def auto_finetune_loop():
    global is_finetuning, model, optimizer, bpe
    while True:
        time.sleep(300)
        with train_data_lock:
            data = load_train_data()
            if len(data) >= fine_tune_threshold and not is_finetuning:
                logging.info("Автоматический запуск fine-tuning по накоплению данных")
                is_finetuning = True
                try:
                    texts = [pair[0] for pair in data] + [pair[1] for pair in data]

                    sentences = []
                    for text in texts:
                        sentences.extend(split_text_to_sentences(text))
                    sentences = sentences[:50]

                    fine_tune(model, optimizer, bpe, sentences, epochs=10)

                    data.clear()
                    save_train_data(data)
                    logging.info("Автоматический fine-tuning завершён")
                except Exception as e:
                    logging.error(f"Ошибка во время автоматического fine-tuning: {e}")
                finally:
                    is_finetuning = False

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "Привет! Я Qsusli — твой AI-ассистент.\n\n"
        "Вот что я умею:\n"
        "🎨 /image <описание> — сгенерировать реалистичное изображение\n"
        "🧠 /brain <текст> — семантический анализ текста\n"
        "⭐ /rate 1-5 — оценить последнее изображение или текстовое сообщение\n"
        "✍️ /correct <текст> — предложить исправленный ответ ИИ для обучения и улучшения\n"
        "🔧 /finetune — дообучить модель на накопленных данных\n"
        "📊 /activity 0-10 — настроить активность в группах\n\n"
        "📝 /train — начать ввод текста для дообучения (многострочный текст поддерживается):\n"
        "    После команды отправляйте текст, когда закончите — напишите /done\n\n"
        "🧹 /clear — очистить текущий контекст диалога\n\n"
        "Если нужна помощь, пиши /help"
    )
    safe_send_message(message, welcome_text)
    logging.info(f"Пользователь {message.from_user.id} использовал /start")

@bot.message_handler(commands=['clear', 'reset', 'clearcontext', 'clear_context', 'resetcontext', 'reset_context'])
def clear_context(message):
    chat_id = message.chat.id
    workspace = workspaces.get(chat_id)
    if workspace:
        workspace.dialog_history.clear()
        workspace.context_keywords.clear()
        workspace.semantic_info = None
        workspace.current_prompt = None
        bot.reply_to(message, "Контекст очищен.")
    else:
        bot.reply_to(message, "Контекст для этого чата не найден.")

@bot.message_handler(commands=['activity'])
def set_activity(message):
    global activity_settings

    # Проверяем, что команда вызывается в группе или супергруппе
    if message.chat.type not in ['group', 'supergroup']:
        bot.reply_to(message, "❌ Команда /activity доступна только в группах.")
        return

    # Проверяем права администратора
    try:
        member = bot.get_chat_member(message.chat.id, message.from_user.id)
        if member.status not in ['administrator', 'creator']:
            bot.reply_to(message, "❌ Только администраторы могут изменять активность бота")
            return
    except Exception as e:
        bot.reply_to(message, "❌ Не удалось проверить права администратора")
        return

    try:
        # Извлекаем уровень активности из текста команды
        activity_text = message.text[10:].strip()  # Убираем "/activity "

        if not activity_text:
            current_level = activity_settings.get(str(message.chat.id), 5)
            bot.reply_to(message,
                f"📊 Текущий уровень активности: {current_level}/10\n\n"
                "0 = отвечаю только при ответе на мои сообщения\n"
                "5 = средняя активность (по умолчанию)\n"
                "10 = отвечаю на все сообщения\n\n"
                "Используйте: /activity <0-10>"
            )
            return

        try:
            activity_level = int(activity_text)
            if activity_level < 0 or activity_level > 10:
                bot.reply_to(message, "❌ Уровень активности должен быть от 0 до 10")
                return
        except ValueError:
            bot.reply_to(message, "❌ Уровень активности должен быть числом от 0 до 10")
            return

        # Сохраняем настройку в словарь и базу
        activity_settings[str(message.chat.id)] = activity_level
        db.set_group_setting(message.chat.id, 'activity_level', activity_level)

        # Описываем поведение для пользователя
        behavior_desc = {
            0: "отвечаю только при ответе на мои сообщения",
            1: "очень низкая активность (5%)",
            2: "низкая активность (10%)",
            3: "пониженная активность (20%)",
            4: "умеренная активность (35%)",
            5: "средняя активность (50%)",
            6: "повышенная активность (65%)",
            7: "высокая активность (80%)",
            8: "очень высокая активность (90%)",
            9: "максимальная активность (95%)",
            10: "отвечаю на все сообщения (100%)"
        }

        bot.reply_to(message,
            f"✅ Уровень активности установлен: {activity_level}/10\n"
            f"📝 Поведение: {behavior_desc[activity_level]}"
        )

        logging.info(f"Активность в чате {message.chat.id} установлена на {activity_level}")

    except Exception as e:
        error_msg = f"Ошибка при установке активности: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        bot.reply_to(message, "❌ Произошла ошибка при установке активности")

@bot.message_handler(commands=['finetune'])
def start_finetune(message):
    global is_finetuning
    with train_data_lock:
        if is_finetuning:
            bot.reply_to(message, "Дообучение уже запущено, подождите.")
            return
        data = db.load_train_data()
        if not data:
            bot.reply_to(message, "Нет новых данных для дообучения.")
            return
        bot.reply_to(message, "Начинаю дообучение модели на новых данных...")
        is_finetuning = True

    try:
        texts = [pair[0] for pair in data] + [pair[1] for pair in data]
        sentences = []
        for t in texts:
            sentences.extend(split_text_to_sentences(t))
        sentences = sentences[:50]

        fine_tune(model, optimizer, bpe, sentences, epochs=10)

        with train_data_lock:
            db.clear_train_data()
        bot.reply_to(message, "Дообучение завершено и данные очищены.")
    except Exception as e:
        bot.reply_to(message, f"Ошибка во время дообучения: {e}")
        logging.error(f"Ошибка при manual fine-tuning: {e}")
    finally:
        is_finetuning = False

@bot.message_handler(commands=['train'])
def start_train_dialog(message):
    chat_id = message.chat.id
    user_training_buffers[chat_id] = []
    bot.reply_to(message, "Отправьте текст для обучения. Когда закончите — напишите /done.")

@bot.message_handler(commands=['done'])
def finish_train_dialog(message):
    chat_id = message.chat.id
    if chat_id not in user_training_buffers or not user_training_buffers[chat_id]:
        bot.reply_to(message, "Вы не начали ввод текста для обучения. Напишите /train чтобы начать.")
        return

    text = '\n'.join(user_training_buffers[chat_id])
    del user_training_buffers[chat_id]

    bot.reply_to(message, "Начинаю дообучение на вашем тексте...")

    sentences = split_text_to_sentences(text)
    sentences = sentences[:500]

    def train_thread():
        global is_finetuning
        is_finetuning = True
        try:
            fine_tune(model, optimizer, bpe, sentences, epochs=10)
            bot.send_message(chat_id, "Дообучение завершено.")
        except Exception as e:
            bot.send_message(chat_id, f"Ошибка при дообучении: {e}")
        finally:
            is_finetuning = False

    threading.Thread(target=train_thread, daemon=True).start()

@bot.message_handler(func=lambda m: m.chat.id in user_training_buffers)
def collect_training_text(message):
    user_training_buffers[message.chat.id].append(message.text)

@bot.message_handler(commands=['image'])
def generate_image(message):
    try:
        prompt = message.text[7:].strip()
        print(f"[DEBUG] Получен запрос на изображение: '{prompt}'")
        if not prompt:
            bot.reply_to(message, "Пожалуйста, укажите описание изображения после команды /image")
            return

        bot.reply_to(message, f"🎨 Генерирую изображение для: '{prompt}'...")

        generated_image_path = generate_image_from_text(prompt, bpe, max_images=5)
        print(f"[DEBUG] generate_image_from_text вернула: {generated_image_path}")

        if generated_image_path and os.path.exists(generated_image_path):
            print(f"[DEBUG] Файл изображения существует: {generated_image_path}")
            with open(generated_image_path, 'rb') as photo:
                caption = f"🖼️ Изображение: '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'\n\n💬 Оцените командой /rate [1-5]"
                bot.send_photo(message.chat.id, photo, caption=caption)

            user_last_outputs[message.chat.id] = {
                'type': 'image',
                'query': prompt,
                'response': generated_image_path
            }

            db.insert_into('image_generations', prompt=prompt, image_path=generated_image_path, timestamp=str(time.time()))

            return

        else:
            print("[DEBUG] Файл изображения не найден или путь пуст")
            bot.reply_to(message, "❌ Произошла ошибка при генерации изображения.")

    except Exception as e:
        logging.error(f"Ошибка генерации изображения: {e}")
        traceback.print_exc()
        bot.reply_to(message, "❌ Произошла ошибка при обработке изображения.")

@bot.message_handler(commands=['brain'])
def analyze_brain(message):
    try:
        # Извлекаем текст для анализа
        text_to_analyze = message.text[7:].strip()
        
        if not text_to_analyze:
            bot.reply_to(message, "Пожалуйста, укажите текст для анализа после команды /brain")
            return
        
        bot.reply_to(message, f"🧠 Анализирую текст: '{text_to_analyze[:50]}{'...' if len(text_to_analyze) > 50 else ''}'")
        
        # Семантический анализ
        from qsusli_model import analyze_semantic_meaning, get_context_keywords
        
        semantic_data = analyze_semantic_meaning(text_to_analyze)
        keywords = get_context_keywords(text_to_analyze)
        
        if semantic_data:
            result = f"📊 **Семантический анализ:**\n"
            result += f"🎯 Основная категория: {semantic_data['primary_category']}\n"
            result += f"📝 Все категории: {', '.join(semantic_data['all_categories'])}\n"
            result += f"🎚️ Уверенность: {semantic_data['confidence']:.1%}\n"
            result += f"📏 Длина: {semantic_data['text_length']} слов\n"
            result += f"🧩 Сложность: {semantic_data['complexity']}\n"
            
            if keywords:
                result += f"🔑 Ключевые слова: {', '.join(keywords[:5])}\n"
            
            # Добавляем рекомендации
            if semantic_data['primary_category'] == 'вопрос':
                result += "\n💡 **Рекомендация:** Это вопрос - можно искать информацию в Wikipedia"
            elif semantic_data['primary_category'] == 'эмоция':
                result += "\n💡 **Рекомендация:** Эмоциональный контент - подходит для творческого ответа"
            elif semantic_data['primary_category'] == 'действие':
                result += "\n💡 **Рекомендация:** Описание действия - можно генерировать пошаговые инструкции"
            
            bot.reply_to(message, result)
        else:
            bot.reply_to(message, "❌ Не удалось провести семантический анализ")
            
        logging.info(f"Пользователь {message.from_user.id} использовал brain4 анализ для: '{text_to_analyze}'")
        
    except Exception as e:
        error_msg = f"Ошибка при анализе мозга: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        bot.reply_to(message, "❌ Произошла ошибка при анализе")


@bot.message_handler(commands=['rate'])
def rate_response(message):
    try:
        rating_text = message.text[6:].strip()
        if not rating_text:
            bot.reply_to(message, "Пожалуйста, укажите оценку от 1 до 5 после команды /rate")
            return

        try:
            rating = int(rating_text)
            if rating < 1 or rating > 5:
                bot.reply_to(message, "Оценка должна быть от 1 до 5")
                return
        except ValueError:
            bot.reply_to(message, "Оценка должна быть числом от 1 до 5")
            return

        last_output = user_last_outputs.get(message.chat.id)
        print(f"Оцениваемый объект: {last_output}")
        if not last_output:
            bot.reply_to(message, "Сначала получите ответ или изображение от бота, чтобы его оценить.")
            return

        user_id = message.from_user.id
        query = last_output.get('query', '')
        response = last_output.get('response', '')

        if last_output.get('type') == 'image':
            from qsusli_model import evaluate_image_quality, reinforcement_learning_update, improve_unknown_word_generation

            normalized_rating = (rating - 1) / 4.0

            quality = evaluate_image_quality(response, user_rating=normalized_rating)

            category = improve_unknown_word_generation(query) or 'default'
            generation_params = {
                'colors': [],  # Можно извлечь из метаданных изображения
                'shapes': '',
                'query': query
            }

            reinforcement_learning_update(category, normalized_rating, generation_params)
            print(f"🧠 RLHF: Обновлена политика для '{category}' с оценкой {normalized_rating:.3f}")

            db.insert_into(
                'image_ratings',
                user_id=user_id,
                query=query,
                image_path=response,
                rating=rating,
                normalized_rating=normalized_rating,
                auto_score=quality.get('auto_score', 0.5),
                timestamp=str(time.time())
            )

            emoji_map = {1: "😞", 2: "😐", 3: "🙂", 4: "😊", 5: "🤩"}
            bot.reply_to(message, f"Спасибо за оценку! {emoji_map[rating]} Ваша оценка: {rating}/5\n"
                                  f"Автоматическая оценка: {quality['auto_score']:.2f}\n"
                                  f"Эта информация поможет улучшить качество генерации изображений!")

        elif last_output.get('type') == 'text':
            db.insert_into(
                'text_ratings',
                user_id=user_id,
                query=query,
                response=response,
                rating=rating,
                timestamp=str(time.time())
            )
            bot.reply_to(message, f"Спасибо за оценку текста {rating}! Это поможет улучшить меня.")
        else:
            bot.reply_to(message, "Неизвестный тип последнего ответа для оценки.")

    except Exception as e:
        logging.error(f"Ошибка при обработке /rate: {e}")
        traceback.print_exc()
        bot.reply_to(message, "❌ Произошла ошибка при сохранении оценки.")


@bot.message_handler(commands=['correct'])
def handle_correct(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    corrected_text = message.text[len('/correct'):].strip()

    last_output = user_last_outputs.get(chat_id)
    if not last_output or last_output.get('type') != 'text':
        bot.reply_to(message, "Нет последнего текстового ответа для корректировки.")
        return

    original_query = last_output.get('query')
    original_response = last_output.get('response')

    if not corrected_text:
        bot.reply_to(message, "Пожалуйста, укажите исправленный вариант после команды.")
        return

    db.save_correction(user_id=user_id, query=original_query, original_response=original_response, corrected_response=corrected_text, timestamp=time.time())

    threading.Thread(target=process_rlhf_correction, args=(original_query, original_response, corrected_text), daemon=True).start()

    bot.reply_to(message, "Спасибо! Ваш исправленный ответ учтён для улучшения модели.")


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    global is_finetuning

    if is_finetuning:
        bot.reply_to(message, "Сейчас идет дообучение, попробуйте позже.")
        return

    user_text = message.text
    if not user_text or not user_text.strip():
        bot.reply_to(message, "Пустое сообщение, попробуйте ещё раз.")
        return

    user_text_lower = user_text.lower()

    # Получаем или создаём рабочее пространство для текущего чата
    workspace = workspaces.get(message.chat.id)
    if not workspace:
        from qsusli_model import GlobalWorkspace
        workspace = GlobalWorkspace()
        workspaces[message.chat.id] = workspace

    # Очистка контекста по команде
    if user_text_lower.strip() in ['/clearcontext', '/resetcontext']:
        workspace.dialog_history.clear()
        workspace.context_keywords.clear()
        workspace.semantic_info = None
        workspace.current_prompt = None
        bot.reply_to(message, "Контекст очищен.")
        return

    # Обновляем семантический контекст и ключевые слова
    try:
        from qsusli_model import analyze_semantic_meaning, get_context_keywords, improve_unknown_word_generation, get_rlhf_params_for_category
        semantic_context = analyze_semantic_meaning(user_text_lower)
        context_keywords = get_context_keywords(user_text_lower)
        workspace.update_semantic(semantic_context)
        workspace.update_keywords(context_keywords)
        workspace.set_current_prompt(user_text_lower)

        category = improve_unknown_word_generation(user_text_lower) or 'default'
        rlhf_params = get_rlhf_params_for_category(category)
    except Exception as e:
        logging.error(f"Ошибка семантического анализа или RLHF: {e}")
        category = 'default'
        rlhf_params = {}

    # Логика определения, нужно ли отвечать, с подробной обработкой групп
    should_respond = True
    if message.chat.type in ['group', 'supergroup']:
        activity_level = activity_settings.get(str(message.chat.id), 2)  # Значение по умолчанию

        if activity_level == 0:
            # Отвечаем только если сообщение — ответ на сообщение бота
            should_respond = False
            if message.reply_to_message and message.reply_to_message.from_user.is_bot:
                should_respond = True
                logging.info(f"Группа {message.chat.id}: ответ на сообщение бота (активность 0)")

        else:
            base_chance = max(0.05, activity_level / 20.0)  # Немного пониженный базовый шанс
            response_chance = base_chance

            if len(user_text_lower.strip()) <= 2:
                should_respond = False
                logging.info(f"Группа {message.chat.id}: пропуск короткого сообщения '{user_text_lower}'")
            else:
                if any(word in user_text_lower for word in ['qsusli', 'бот', 'bot', '@']):
                    response_chance = min(0.8, base_chance + 0.4)
                elif message.reply_to_message and message.reply_to_message.from_user.is_bot:
                    response_chance = min(0.9, base_chance + 0.6)
                elif any(word in user_text_lower for word in ['что', 'как', 'где', 'когда', 'почему', 'зачем', '?']):
                    response_chance = min(0.6, base_chance + 0.2)
                elif any(word in user_text_lower for word in ['привет', 'hi', 'hello', 'помощь', 'help']):
                    response_chance = min(0.7, base_chance + 0.3)

                should_respond = random.random() < response_chance

                if should_respond:
                    logging.info(f"Группа {message.chat.id}: активность {activity_level}/10, шанс {response_chance:.2f}, отвечаем")

    generated_text = "Без ответа"
    if should_respond:
        try:
            full_tokens = bpe.encode(user_text_lower)
            start_tokens = tokens_to_indices(full_tokens[:3], bpe.vocab)
            if not start_tokens or all(token == bpe.vocab.get('<unk>', 0) for token in start_tokens):
                for word in user_text_lower.split():
                    word_tokens = tokens_to_indices(bpe.encode(word), bpe.vocab)
                    if word_tokens and not all(token == bpe.vocab.get('<unk>', 0) for token in word_tokens):
                        start_tokens = word_tokens[:2]
                        break
                if not start_tokens or all(token == bpe.vocab.get('<unk>', 0) for token in start_tokens):
                    start_tokens = [list(bpe.vocab.values())[torch.randint(0, len(bpe.vocab), (1,)).item()]]

            generation_params = {
                'temperature': rlhf_params.get('temperature', 0.7),
                'max_len': rlhf_params.get('max_len', 25),
                'top_k': rlhf_params.get('top_k', 50),
                'top_p': rlhf_params.get('top_p', 0.9)
            }

            generation_params = workspace.meta_learning_update(generation_params)

            generated_tokens = generate_text(
                model,
                start_tokens,
                max_len=generation_params['max_len'],
                inv_vocab=inv_vocab,
                temperature=generation_params['temperature'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p']
            )

            text_parts = []
            for token in generated_tokens:
                if token == '<space>':
                    text_parts.append(' ')
                elif token.endswith('</w>'):
                    text_parts.append(token.replace('</w>', ''))
                    text_parts.append(' ')
                elif token != '<unk>':
                    text_parts.append(token)
            generated_text = ''.join(text_parts).strip()

            if len(generated_text) < 3 or generated_text.count(' ') < 1:
                generated_tokens = generate_text(
                    model,
                    start_tokens,
                    max_len=15,
                    inv_vocab=inv_vocab,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.9
                )
                text_parts = []
                for token in generated_tokens:
                    if token == '<space>':
                        text_parts.append(' ')
                    elif token.endswith('</w>'):
                        text_parts.append(token.replace('</w>', ''))
                        text_parts.append(' ')
                    elif token != '<unk>':
                        text_parts.append(token)
                generated_text = ''.join(text_parts).strip()

            if not generated_text or len(generated_text) < 2:
                generated_text = "Интересно..."

            success = safe_send_message(message, generated_text)
            if not success:
                logging.warning(f"Failed to send message to chat {message.chat.id}: '{generated_text[:50]}...'")
                generated_text = "Сообщение не отправлено из-за лимитов"

            workspace.add_dialog_pair(user_text_lower, generated_text)

            user_last_outputs[message.chat.id] = {
                'type': 'text',
                'query': user_text_lower,
                'response': generated_text
            }

            with train_data_lock:
                db.insert_into('train_data', input_text=user_text_lower, output_text=generated_text, timestamp=str(time.time()))

        except Exception as e:
            logging.error(f"Ошибка генерации ответа: {e}")
            safe_send_message(message, "Произошла ошибка при генерации ответа.")

    # Добавление предложений из Wikipedia для обучения
    with train_data_lock:
        try:
            if should_respond and len(user_text_lower) > 5 and random.random() < 0.3:
                wiki_sentences = search_wikipedia(user_text_lower, sentences_limit=2)
                if wiki_sentences:
                    for sent in wiki_sentences:
                        db.insert_into('train_data', input_text=sent, output_text=sent, timestamp=str(time.time()))
                    logging.info(f"Wikipedia: добавлено {len(wiki_sentences)} предложений")
        except Exception as e:
            logging.warning(f"Wikipedia search error: {e}")


if __name__ == '__main__':
    try:
        threading.Thread(target=auto_finetune_loop, daemon=True).start()
        logging.info("Бот запущен...")
        print("Бот запущен...")
        
        while True:
            try:
                bot.polling(none_stop=True, interval=1, timeout=60)
            except Exception as e:
                print(f"Ошибка polling: {e}")
                logging.error(f"Ошибка polling: {e}")
                time.sleep(5)  # Пауза перед перезапуском
                print("Перезапускаю polling...")
                
    except KeyboardInterrupt:
        print("Бот остановлен пользователем")
        logging.info("Бот остановлен пользователем")
    except Exception as e:
        print(f"Критическая ошибка бота: {e}")
        logging.error(f"Критическая ошибка бота: {e}")
        traceback.print_exc()
