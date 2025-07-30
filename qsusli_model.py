import re
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import urllib.parse
import urllib.request
import hashlib
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from collections import defaultdict
import time
import random
import math
from db import DB

db = DB()

class BPE:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def get_stats(self, tokens):
        pairs = defaultdict(int)
        for token, freq in tokens.items():
            symbols = token.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, tokens):
        new_tokens = {}
        bigram = ' '.join(pair)
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        for token, freq in tokens.items():
            new_token = pattern.sub(''.join(pair), token)
            new_tokens[new_token] = freq + new_tokens.get(new_token, 0)
        return new_tokens

    def learn_bpe(self, corpus, max_merges=3000, min_freq=10):
        tokens = {}
        # Собираем базовые токены из корпуса
        base_chars = set()
        for word in corpus:
            base_chars.update(list(word))
            token = ' '.join(list(word) + ['</w>'])
            tokens[token] = tokens.get(token, 0) + 1

        # Добавляем базовые символы явно в словарь
        self.vocab = {ch: idx for idx, ch in enumerate(sorted(base_chars))}
        self.vocab.update({'<unk>': len(self.vocab), '<pad>': len(self.vocab), '</w>': len(self.vocab), '<space>': len(self.vocab)})

        for i in range(max_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best_pair, best_freq = max(pairs.items(), key=lambda x: x[1])
            if best_freq < min_freq:
                break
            tokens = self.merge_vocab(best_pair, tokens)
            self.merges.append(best_pair)
            # Обновляем словарь новыми подсловами
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

    def encode(self, text):
        if not text or not text.strip():
            return ['<unk>']

        text = text.lower().strip()
        words = text.split()
        all_tokens = []

        for i, word in enumerate(words):
            if i > 0:
                all_tokens.append('<space>')
            word_tokens = list(word) + ['</w>']

            for merge in self.merges:
                j = 0
                while j < len(word_tokens) -1:
                    if word_tokens[j] == merge[0] and word_tokens[j+1] == merge[1]:
                        word_tokens[j:j+2] = [''.join(merge)]
                    else:
                        j +=1

            all_tokens.extend(word_tokens)

        result = []
        for token in all_tokens:
            if token in self.vocab:
                result.append(token)
            else:
                result.append('<unk>')

        return result

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.max_len = max_len

    def forward(self, src, src_key_padding_mask=None):
        seq_len = src.size(1)
        if seq_len > self.max_len:
            src = src[:, :self.max_len]
            seq_len = self.max_len
        emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        out = self.fc_out(out)
        return out

class GlobalWorkspace:
    def __init__(self):
        self.semantic_info = None
        self.context_keywords = []
        self.user_ratings = []
        self.dialog_history = []
        self.current_prompt = None

    def update_semantic(self, semantic_data):
        self.semantic_info = semantic_data

    def update_keywords(self, keywords):
        self.context_keywords = keywords

    def add_user_rating(self, rating):
        self.user_ratings.append(rating)
        if len(self.user_ratings) > 50:
            self.user_ratings = self.user_ratings[-50:]

    def add_dialog_pair(self, user_text, bot_response):
        self.dialog_history.append((user_text, bot_response))
        if len(self.dialog_history) > 100:
            self.dialog_history = self.dialog_history[-100:]

    def set_current_prompt(self, prompt):
        self.current_prompt = prompt

    def summarize_state(self):
        return {
            'primary_category': self.semantic_info.get('primary_category') if self.semantic_info else None,
            'keywords': self.context_keywords,
            'recent_ratings_avg': (sum(self.user_ratings)/len(self.user_ratings)) if self.user_ratings else None,
            'last_user_message': self.dialog_history[-1][0] if self.dialog_history else None
        }

    def meta_learning_update(self, generation_params=None, success_metric=None):
        if generation_params is None:
            generation_params = {}

        avg_rating = self.user_ratings[-10:]
        if avg_rating:
            avg = sum(avg_rating)/len(avg_rating)
            if avg < 0.5:
                generation_params['temperature'] = max(0.3, generation_params.get('temperature', 0.7) - 0.1)
            elif avg > 0.8:
                generation_params['temperature'] = min(1.0, generation_params.get('temperature', 0.7) + 0.1)

        return generation_params

def tokens_to_indices(tokens, vocab):
    unk_idx = vocab.get('<unk>', 0)
    return [vocab.get(token, unk_idx) for token in tokens if token]

def pad_sequences(sequences, pad_value=0, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padded = seq + [pad_value] * (max_len - len(seq))
        padded_seqs.append(padded)
    return padded_seqs

def prepare_sequences(sentences, bpe):
    inputs = []
    targets = []
    for sentence in sentences:
        words = sentence.lower().split()
        tokens = []
        for w in words:
            tokens.extend(bpe.encode(w))
        if len(tokens) < 2:
            continue
        inputs.append(tokens_to_indices(tokens[:-1], bpe.vocab))
        targets.append(tokens_to_indices(tokens[1:], bpe.vocab))
    if not inputs:
        return None, None, None
    max_len = max(max(len(seq) for seq in inputs), max(len(seq) for seq in targets))
    inputs_padded = pad_sequences(inputs, pad_value=bpe.vocab.get('<pad>', 0), max_len=max_len)
    targets_padded = pad_sequences(targets, pad_value=bpe.vocab.get('<pad>', 0), max_len=max_len)

    padding_mask = []
    for seq in inputs_padded:
        mask = [token == bpe.vocab.get('<pad>', 0) for token in seq]
        padding_mask.append(mask)

    input_tensor = torch.tensor(inputs_padded, dtype=torch.long)
    target_tensor = torch.tensor(targets_padded, dtype=torch.long)
    padding_mask_tensor = torch.tensor(padding_mask)  # bool или byte

    return input_tensor, target_tensor, padding_mask_tensor

def split_text_to_sentences(text, max_len=50):
    # Сначала пытаемся разбить по знакам препинания
    sentences = re.split(r'[.!?]+', text)
    filtered = []
    for sent in sentences:
        sent = sent.strip()
        if 5 <= len(sent.split()) <= max_len:
            filtered.append(sent)

    # Если не нашли подходящих предложений, разбиваем по строкам
    if not filtered:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 1 <= len(line.split()) <= max_len:
                filtered.append(line)

    return filtered

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # logits - одномерный тензор
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Удаляем токены с кумулятивной вероятностью выше top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Сдвигаем на 1, чтобы оставить хотя бы один токен
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def generate_text(model, start_tokens, max_len, inv_vocab, temperature=1.0, top_k=50, top_p=0.9, bpe_vocab=None):
    model.eval()
    generated = start_tokens[:]
    input_seq = torch.tensor([start_tokens])
    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_seq)
            logits = output[0, -1] / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            if bpe_vocab is not None:
                space_token_idx = bpe_vocab.get('<space>')
                if space_token_idx is not None:
                    prob_space = torch.softmax(logits, dim=0)[space_token_idx].item()
                    print(f"Вероятность пробела: {prob_space:.4f}")

            probs = torch.softmax(logits, dim=0)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print("Некорректные вероятности, генерация прервана.")
                break
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_seq = torch.tensor([generated])
            if inv_vocab[next_token].endswith('</w>'):
                break
    return [inv_vocab[idx] for idx in generated]

def train_new_model():
    corpus = [
        # English greetings and common phrases
        "hello world", "hi there", "hey you", "greetings friend", "good morning everyone",
        "good evening all", "how are you today", "what's up", "how is it going",
        "nice to meet you", "thank you very much", "thanks a lot", "you're welcome",
        "sorry about that", "excuse me please", "goodbye for now", "see you later",
        "take care", "have a nice day", "yes please", "no thanks", "maybe later",
        "I don't know", "can you help me", "what is your name", "tell me a joke",
        "how old are you", "where are you from", "what do you do", "I love programming",
        "Python is great", "let's learn AI", "machine learning is fun",
        "artificial intelligence is fascinating", "neural networks are powerful",
        "deep learning techniques", "have a good evening", "see you soon",
        "please wait a moment", "thank you for your help", "sorry for the delay",
        "can you explain that", "what time is it", "how can I assist you", "good night",
        "happy birthday", "congratulations", "let's start", "welcome aboard",
        "have a great weekend", "enjoy your meal", "best wishes", "all the best",

        # Russian greetings and common phrases
        "привет мир", "здравствуй друг", "доброе утро всем", "добрый вечер", "как дела сегодня",
        "что нового", "рад тебя видеть", "спасибо большое", "пожалуйста", "извини за это",
        "простите пожалуйста", "до свидания", "увидимся позже", "береги себя", "хорошего дня",
        "да, пожалуйста", "нет, спасибо", "может позже", "я не знаю", "ты можешь помочь",
        "как тебя зовут", "расскажи шутку", "сколько тебе лет", "откуда ты", "чем занимаешься",
        "я люблю программировать", "Python отличный язык", "давай изучать ИИ", "машинное обучение интересно",
        "искусственный интеллект удивителен", "нейронные сети мощные", "техники глубокого обучения",
        "хорошего вечера", "до скорого", "подожди минуту", "спасибо за помощь", "извини за задержку",
        "можешь объяснить это", "который час", "чем могу помочь", "спокойной ночи",
        "с днем рождения", "поздравляю", "давай начнем", "добро пожаловать на борт",
        "хороших выходных", "приятного аппетита", "лучшие пожелания", "всего наилучшего",

        # English questions and info
        "what is the weather today", "how to cook pasta", "tell me a story",
        "what is the capital of France", "how do airplanes fly", "what is quantum physics",
        "define artificial intelligence", "explain machine learning", "who is Elon Musk",
        "latest news in technology", "how to learn programming", "tips for studying",
        "best movies of 2025", "how to stay healthy", "what is meditation",
        "history of the internet", "how to play chess", "what is blockchain",
        "how to improve memory", "what is climate change", "how to travel cheap",
        "best books to read", "how to start a business", "what is cryptocurrency",
        "how to bake a cake", "what is the meaning of life", "how to be happy",
        "tips for time management", "best programming languages", "how to learn Python",
        "what is deep learning", "how to build a website", "what is data science",
        "how to make coffee", "best places to visit", "how to learn English",
        "what is virtual reality", "how to write a resume", "how to meditate",
        "best exercises for abs", "how to lose weight", "what is AI ethics",
        "how to improve coding skills", "best podcasts to listen", "how to invest money",
        "what is natural language processing", "how to start blogging", "best hiking trails",
        "how to learn guitar", "what is cloud computing", "how to create art",
        "best ways to relax", "how to speak in public", "what is 5G technology",

        # Russian questions and info
        "какая сегодня погода", "как приготовить пасту", "расскажи мне историю",
        "какая столица Франции", "как летают самолеты", "что такое квантовая физика",
        "определи искусственный интеллект", "объясни машинное обучение", "кто такой Илон Маск",
        "последние новости технологий", "как учить программирование", "советы для учебы",
        "лучшие фильмы 2025 года", "как оставаться здоровым", "что такое медитация",
        "история интернета", "как играть в шахматы", "что такое блокчейн",
        "как улучшить память", "что такое изменение климата", "как путешествовать дешево",
        "лучшие книги для чтения", "как начать бизнес", "что такое криптовалюта",
        "как испечь торт", "в чем смысл жизни", "как быть счастливым",
        "советы по тайм-менеджменту", "лучшие языки программирования", "как учить Python",
        "что такое глубокое обучение", "как создать сайт", "что такое наука о данных",
        "как приготовить кофе", "лучшие места для посещения", "как учить английский",
        "что такое виртуальная реальность", "как написать резюме", "как медитировать",
        "лучшие упражнения для пресса", "как похудеть", "что такое этика ИИ",
        "как улучшить навыки кодирования", "лучшие подкасты для прослушивания", "как инвестировать деньги",
        "что такое обработка естественного языка", "как начать блог", "лучшие маршруты для походов",
        "как учить гитару", "что такое облачные вычисления", "как создавать искусство",
        "лучшие способы расслабиться", "как говорить публично", "что такое технология 5G",

        # English emotions and moods
        "I am happy today", "feeling sad", "excited about the trip", "worried about exams",
        "love and friendship", "feeling tired", "full of energy", "need a break",

        # Russian emotions and moods
        "я сегодня счастлив", "чувствую грусть", "взволнован поездкой", "беспокоюсь из-за экзаменов",
        "любовь и дружба", "чувствую усталость", "полон энергии", "нужен перерыв",

        # English commands and phrases
        "open the door", "turn on the lights", "play some music", "set an alarm for 7 am",
        "call mom", "send an email", "what's the news", "tell me a joke",
        "translate this sentence", "find nearby restaurants", "book a taxi",
        "order pizza", "check the weather", "remind me to buy milk",
        "how to fix a bike", "what is the time", "where is the nearest bank",
        "how to learn cooking", "best programming tutorials", "latest tech trends",
        "how to meditate properly", "tips for good sleep", "how to reduce stress",
        "best travel destinations", "how to learn languages fast", "how to write poetry",
        "how to improve memory", "how to be productive", "healthy lifestyle tips",

        # Russian commands and phrases
        "открой дверь", "включи свет", "включи музыку", "установи будильник на 7 утра",
        "позвони маме", "отправь письмо", "какие новости", "расскажи шутку",
        "переведи это предложение", "найди ближайшие рестораны", "закажи такси",
        "закажи пиццу", "проверь погоду", "напомни купить молоко",
        "как починить велосипед", "который час", "где ближайший банк",
        "как научиться готовить", "лучшие уроки программирования", "последние тенденции технологий",
        "как правильно медитировать", "советы для хорошего сна", "как снизить стресс",
        "лучшие туристические направления", "как быстро учить языки", "как писать стихи",
        "как улучшить память", "как быть продуктивным", "советы для здорового образа жизни",

        # Additional English phrases with punctuation and spaces
        "Hello, how are you?", "What's your name?", "Can you help me, please?",
        "I don't know what to do.", "Let's meet at 5 pm.", "Thank you very much!",
        "Sorry, I was late.", "Good morning, everyone!", "Have a nice day!",
        "See you soon.", "Take care!", "How's the weather today?", "Is it raining?",
        "Where can I find a good restaurant?", "Tell me something interesting.",
        "Do you like music?", "I love programming in Python.", "Let's learn AI together.",
        "Can you explain machine learning?", "What is deep learning?", "How old are you?",
        "What time is it now?", "Please help me with my homework.", "Happy birthday!",
        "Congratulations on your achievement!", "Good luck!", "Best wishes to you.",

        # Additional Russian phrases with punctuation and spaces
        "Привет, как дела?", "Как тебя зовут?", "Можешь помочь, пожалуйста?",
        "Я не знаю, что делать.", "Давай встретимся в 5 вечера.", "Большое спасибо!",
        "Извини, я опоздал.", "Доброе утро, всем!", "Хорошего дня!",
        "До скорого.", "Береги себя!", "Какая сегодня погода?", "Идёт ли дождь?",
        "Где найти хороший ресторан?", "Расскажи что-нибудь интересное.",
        "Ты любишь музыку?", "Я люблю программировать на Python.", "Давай учить ИИ вместе.",
        "Можешь объяснить машинное обучение?", "Что такое глубокое обучение?", "Сколько тебе лет?",
        "Который сейчас час?", "Помоги с домашним заданием, пожалуйста.", "С днём рождения!",
        "Поздравляю с достижением!", "Удачи!", "Всего наилучшего."
    ]

    bpe = BPE()
    # Учим BPE с динамическим числом слияний (до 2000, минимум частоты 5)
    bpe.learn_bpe(corpus, max_merges=2000, min_freq=5)

    # Убедимся, что базовые токены есть
    for token in ['<unk>', '<pad>', '</w>', '<space>']:
        if token not in bpe.vocab:
            bpe.vocab[token] = len(bpe.vocab)

    model = Transformer(len(bpe.vocab), embed_size=256, nhead=8, num_layers=6, dim_feedforward=1024)

    # Инициализация весов
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=bpe.vocab.get('<pad>', 0))

    sentences = [
        # English sentences
        "hello world this is a simple example of text data",
        "heaven is often described as a place of peace and happiness",
        "heavy rain can cause flooding in many areas",
        "the hell of war is something no one wants to experience",
        "hello again lets learn how to build ai models",
        "machine learning is transforming many industries",
        "natural language processing enables computers to understand human language",
        "deep neural networks have revolutionized computer vision",
        "reinforcement learning helps agents learn from interaction",
        "python is a popular programming language for AI",
        "data science combines statistics and computer science",
        "artificial intelligence is a broad field of study",
        "supervised learning requires labeled data",
        "unsupervised learning finds hidden patterns",
        "generative models can create realistic images",
        "convolutional neural networks are good for images",
        "recurrent neural networks model sequences",
        "transformers have changed the NLP landscape",
        "training large models requires significant resources",
        "optimization algorithms improve model performance",
        "regularization techniques prevent overfitting",
        "feature engineering is crucial for classical ML",
        "big data fuels modern AI applications",
        "cloud computing enables scalable training",
        "transfer learning leverages pre-trained models",
        "explainable AI helps interpret model decisions",
        "ethical AI considers fairness and bias",
        "autonomous vehicles rely on AI for navigation",
        "speech recognition converts audio to text",
        "chatbots provide automated customer support",
        "recommendation systems personalize user experience",
        "computer vision powers facial recognition",
        "AI can assist in medical diagnosis",
        "robotics integrates AI with hardware",
        "AI research advances rapidly every year",
        "deep learning models require large datasets",
        "data augmentation improves model robustness",
        "hyperparameter tuning optimizes training",
        "cross-validation assesses model generalization",
        "unsupervised clustering groups similar data",
        "reinforcement learning uses rewards for learning",
        "AI ethics addresses privacy concerns",
        "neural networks mimic brain structures",
        "AI applications include gaming and finance",
        "optimization landscapes can be complex",
        "gradient descent is a common optimizer",
        "activation functions introduce non-linearity",
        "backpropagation updates model weights",
        "batch normalization stabilizes training",
        "dropout reduces overfitting",
        "AI models can be deployed on edge devices",

        # Russian sentences
        "привет мир это простой пример текстовых данных",
        "небо часто описывается как место мира и счастья",
        "сильный дождь может вызвать наводнения во многих районах",
        "ад войны — это то, чего никто не хочет испытать",
        "привет снова давай учиться создавать модели ИИ",
        "машинное обучение трансформирует многие отрасли",
        "обработка естественного языка позволяет компьютерам понимать человеческий язык",
        "глубокие нейронные сети революционизировали компьютерное зрение",
        "обучение с подкреплением помогает агентам учиться через взаимодействие",
        "python — популярный язык программирования для ИИ",
        "наука о данных объединяет статистику и информатику",
        "искусственный интеллект — широкая область исследований",
        "обучение с учителем требует размеченных данных",
        "обучение без учителя находит скрытые закономерности",
        "генеративные модели могут создавать реалистичные изображения",
        "сверточные нейронные сети хороши для обработки изображений",
        "рекуррентные нейронные сети моделируют последовательности",
        "трансформеры изменили ландшафт обработки естественного языка",
        "обучение больших моделей требует значительных ресурсов",
        "алгоритмы оптимизации улучшают производительность моделей",
        "техники регуляризации предотвращают переобучение",
        "инженерия признаков важна для классического машинного обучения",
        "большие данные питают современные приложения ИИ",
        "облачные вычисления обеспечивают масштабируемое обучение",
        "трансферное обучение использует предобученные модели",
        "объяснимый ИИ помогает интерпретировать решения моделей",
        "этический ИИ учитывает справедливость и предвзятость",
        "автономные транспортные средства полагаются на ИИ для навигации",
        "распознавание речи преобразует аудио в текст",
        "чатботы обеспечивают автоматическую поддержку клиентов",
        "рекомендательные системы персонализируют пользовательский опыт",
        "компьютерное зрение поддерживает распознавание лиц",
        "ИИ может помочь в медицинской диагностике",
        "робототехника интегрирует ИИ с аппаратным обеспечением",
        "исследования ИИ быстро развиваются каждый год",
        "глубокие модели обучения требуют больших наборов данных",
        "аугментация данных улучшает устойчивость моделей",
        "настройка гиперпараметров оптимизирует обучение",
        "кросс-валидация оценивает обобщаемость модели",
        "обучение без учителя группирует похожие данные",
        "обучение с подкреплением использует награды для обучения",
        "этика ИИ затрагивает вопросы конфиденциальности",
        "нейронные сети имитируют структуры мозга",
        "применения ИИ включают игры и финансы",
        "ландшафты оптимизации могут быть сложными",
        "градиентный спуск — распространенный оптимизатор",
        "функции активации вводят нелинейность",
        "обратное распространение ошибки обновляет веса модели",
        "пакетная нормализация стабилизирует обучение",
        "дропаут снижает переобучение",
        "модели ИИ могут быть развернуты на периферийных устройствах"
    ]

    inputs, targets, padding_mask = prepare_sequences(sentences, bpe)
    if inputs is None or targets is None or padding_mask is None or inputs.size(0) == 0:
        print("Нет данных для обучения.")
        return None, None, None

    model.train()
    epochs = 50

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(inputs, src_key_padding_mask=padding_mask)
        loss = criterion(output.view(-1, len(bpe.vocab)), targets.view(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Обнаружен NaN/Inf в loss на эпохе {epoch}, прерываю обучение.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, loss: {loss.item():.4f}")

    save_model(model, optimizer, bpe)
    return model, optimizer, bpe

def save_model(model, optimizer, bpe, path='qsusli_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bpe_merges': bpe.merges,
        'bpe_vocab': bpe.vocab
    }, path)
    print("Модель сохранена в", path)

def load_model(path='qsusli_model.pth'):
    checkpoint = torch.load(path)
    bpe = BPE()
    bpe.merges = checkpoint['bpe_merges']
    bpe.vocab = checkpoint['bpe_vocab']
    vocab_size = len(bpe.vocab)
    model = Transformer(vocab_size, embed_size=256, nhead=8, num_layers=6, dim_feedforward=1024)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, bpe

def fine_tune(model, optimizer, bpe, new_sentences, epochs=5):
    inputs, targets, padding_mask = prepare_sequences(new_sentences, bpe)
    if inputs is None or targets is None or padding_mask is None or inputs.size(0) == 0:
        print("Нет данных для дообучения после обработки.")
        return

    criterion = nn.CrossEntropyLoss(ignore_index=bpe.vocab.get('<pad>', 0))
    model.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(inputs, src_key_padding_mask=padding_mask)
        loss = criterion(output.view(-1, len(bpe.vocab)), targets.view(-1))

        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
            print(f"Проблемы с loss на эпохе {epoch+1}: {loss.item()}, пропускаю")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Fine-tuning epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")

    save_model(model, optimizer, bpe)

def search_wikipedia(query, sentences_limit=5):
    """Поиск в Википедии с улучшенной обработкой ошибок"""
    try:
        import requests
        import re
        
        S = requests.Session()
        S.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        URL = "https://ru.wikipedia.org/w/api.php"
        
        # Очищаем запрос от лишних символов
        clean_query = re.sub(r'[^\w\s]', '', query).strip()
        if not clean_query:
            print("Пустой поисковый запрос для Wikipedia")
            return []
        
        PARAMS = {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "format": "json",
            "srlimit": 2,  # Уменьшили для стабильности
            "srinfo": "totalhits",
            "srprop": "snippet"
        }
        
        print(f"🔍 Поиск в Wikipedia: '{clean_query}'")
        
        # Первый запрос - поиск статей
        R = S.get(url=URL, params=PARAMS, timeout=10)
        R.raise_for_status()  # Проверяем статус ответа
        
        DATA = R.json()
        
        if 'query' not in DATA or 'search' not in DATA['query']:
            print("Нет результатов поиска в Wikipedia")
            return []
        
        search_results = DATA['query']['search']
        if not search_results:
            print("Пустые результаты поиска в Wikipedia")
            return []
        
        results = []
        
        for item in search_results:
            try:
                page_title = item.get('title', '')
                if not page_title:
                    continue
                
                print(f"📖 Получаю содержимое страницы: {page_title}")
                
                # Второй запрос - получение содержимого страницы
                PARAMS_PAGE = {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": True,
                    "exintro": True,  # Только введение
                    "exlimit": 1,
                    "titles": page_title,
                    "format": "json"
                }
                
                R2 = S.get(url=URL, params=PARAMS_PAGE, timeout=10)
                R2.raise_for_status()
                
                DATA2 = R2.json()
                
                if 'query' not in DATA2 or 'pages' not in DATA2['query']:
                    continue
                
                pages = DATA2['query']['pages']
                
                for page_id in pages:
                    page_data = pages[page_id]
                    
                    # Проверяем что страница существует
                    if 'missing' in page_data:
                        continue
                    
                    text = page_data.get('extract', '')
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    # Разбиваем на предложения
                    sentences = re.split(r'[.!?]+', text)
                    
                    # Фильтруем и очищаем предложения
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 20 and len(sent) < 200:  # Фильтруем по длине
                            results.append(sent)
                            if len(results) >= sentences_limit:
                                break
                    
                    if len(results) >= sentences_limit:
                        break
                        
                if len(results) >= sentences_limit:
                    break
                    
            except Exception as e:
                print(f"Ошибка при обработке страницы '{item.get('title', '')}': {e}")
                continue
        
        print(f"✅ Найдено {len(results)} предложений из Wikipedia")
        return results[:sentences_limit]
        
    except requests.exceptions.Timeout:
        print("⏰ Таймаут при поиске в Wikipedia")
        return []
    except requests.exceptions.RequestException as e:
        print(f"🌐 Сетевая ошибка Wikipedia: {e}")
        return []
    except Exception as e:
        print(f"❌ Общая ошибка поиска Wikipedia: {e}")
        return []

def analyze_semantic_meaning(text):
    """Анализирует семантическое значение текста"""
    try:
        import re
        
        # Категории для семантического анализа
        categories = {
            'вопрос': ['что', 'как', 'где', 'когда', 'почему', 'зачем', 'кто', 'какой', 'сколько', '?'],
            'эмоция': ['грустно', 'весело', 'радостно', 'печально', 'злой', 'счастливый', 'хорошо', 'плохо'],
            'действие': ['делать', 'идти', 'работать', 'играть', 'учиться', 'смотреть', 'читать', 'писать'],
            'время': ['сегодня', 'вчера', 'завтра', 'сейчас', 'потом', 'скоро', 'давно', 'недавно'],
            'место': ['дом', 'школа', 'работа', 'город', 'страна', 'здесь', 'там', 'туда'],
            'оценка': ['хорошо', 'плохо', 'отлично', 'ужасно', 'нормально', 'супер', 'класс'],
            'приветствие': ['привет', 'здравствуй', 'пока', 'до встречи', 'hello', 'hi', 'bye'],
            'технологии': ['компьютер', 'интернет', 'программа', 'код', 'сайт', 'игра', 'приложение']
        }
        
        text_lower = text.lower()
        detected_categories = []
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_categories.append(category)
        
        # Определяем основную семантику
        if detected_categories:
            return {
                'primary_category': detected_categories[0],
                'all_categories': detected_categories,
                'confidence': len(detected_categories) / len(categories),
                'text_length': len(text.split()),
                'complexity': 'simple' if len(text.split()) <= 5 else 'complex'
            }
        
        return {
            'primary_category': 'general',
            'all_categories': ['general'],
            'confidence': 0.1,
            'text_length': len(text.split()),
            'complexity': 'simple' if len(text.split()) <= 5 else 'complex'
        }
        
    except Exception as e:
        print(f"Ошибка семантического анализа: {e}")
        return None

def get_context_keywords(text):
    """Извлекает ключевые слова для контекста"""
    try:
        import re
        
        # Убираем стоп-слова
        stop_words = ['и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'об', 'что', 'это', 'как', 'но', 'а', 'да', 'нет', 'не', 'я', 'ты', 'он', 'она', 'мы', 'вы', 'они']
        
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Возвращаем топ-5 самых длинных слов
        return sorted(keywords, key=len, reverse=True)[:5]
        
    except Exception as e:
        print(f"Ошибка извлечения ключевых слов: {e}")
        return []

def improve_unknown_word_generation(text):
    """Улучшает генерацию для неизвестных слов используя контекст"""
    try:
        semantic_data = analyze_semantic_meaning(text)
        
        if not semantic_data:
            return None
            
        # Возвращаем семантическую категорию для RLHF
        return semantic_data['primary_category']
        
    except Exception as e:
        print(f"Ошибка улучшения генерации: {e}")
        return None

text_rlhf_policy = {}

def load_text_rlhf_policy():
    global text_rlhf_policy
    try:
        text_rlhf_policy = db.load_text_rlhf_policy()
    except Exception as e:
        print(f"Ошибка загрузки text RLHF политики: {e}")
        text_rlhf_policy = {}

def save_text_rlhf_policy(category, data):
    try:
        db.save_text_rlhf_policy(category, data)
    except Exception as e:
        print(f"Ошибка сохранения text RLHF политики: {e}")

def text_reinforcement_learning_update(category, reward, generation_params):
    global text_rlhf_policy
    try:
        if category not in text_rlhf_policy:
            text_rlhf_policy[category] = {
                'rewards': [],
                'avg_reward': 0.0,
                'count': 0,
                'best_params': generation_params
            }

        text_rlhf_policy[category]['rewards'].append(reward)
        text_rlhf_policy[category]['count'] += 1
        text_rlhf_policy[category]['avg_reward'] = sum(text_rlhf_policy[category]['rewards']) / len(text_rlhf_policy[category]['rewards'])

        if reward > text_rlhf_policy[category]['avg_reward']:
            text_rlhf_policy[category]['best_params'] = generation_params.copy()

        if len(text_rlhf_policy[category]['rewards']) > 100:
            text_rlhf_policy[category]['rewards'] = text_rlhf_policy[category]['rewards'][-100:]

        save_text_rlhf_policy(category, text_rlhf_policy[category])
        print(f"RLHF текст: обновлена политика для '{category}', средняя оценка: {text_rlhf_policy[category]['avg_reward']:.3f}")

    except Exception as e:
        print(f"Ошибка обновления text RLHF: {e}")

def process_rlhf_correction(query, original_response, corrected_response):
    """
    Обрабатывает исправление пользователя для RLHF.
    Вычисляет эмбеддинги оригинального и исправленного ответов,
    оценивает улучшение и обновляет политику RLHF.
    """

    try:
        # Токенизируем и преобразуем в индексы
        orig_tokens = bpe.encode(original_response.lower())
        corr_tokens = bpe.encode(corrected_response.lower())

        orig_indices = torch.tensor([tokens_to_indices(orig_tokens, bpe.vocab)])
        corr_indices = torch.tensor([tokens_to_indices(corr_tokens, bpe.vocab)])

        # Маски паддингов (False — нет паддингов, т.к. короткие последовательности)
        orig_mask = torch.zeros_like(orig_indices, dtype=torch.bool)
        corr_mask = torch.zeros_like(corr_indices, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            # Получаем эмбеддинги из входного слоя (embedding) и усредняем по токенам
            orig_embeds = model.embedding(orig_indices).mean(dim=1)  # (1, embed_size)
            corr_embeds = model.embedding(corr_indices).mean(dim=1)  # (1, embed_size)

            # Вычисляем косинусное сходство между оригиналом и исправлением
            similarity = F.cosine_similarity(orig_embeds, corr_embeds).item()

        # Оценка улучшения: чем ниже similarity, тем больше исправление
        improvement = max(0.0, 1.0 - similarity)  # Значение от 0 до 1

        # Определяем категорию запроса (используем существующую функцию)
        category = improve_unknown_word_generation(query) or 'default'

        # Получаем текущие параметры RLHF для категории
        current_params = get_rlhf_params_for_category(category)

        # Обновляем параметры генерации, например, уменьшая температуру при улучшении
        new_temperature = max(0.3, current_params.get('temperature', 0.7) - improvement * 0.1)

        new_params = current_params.copy()
        new_params['temperature'] = new_temperature

        text_reinforcement_learning_update(category, improvement, new_params)

        print(f"RLHF Correction: категория '{category}', улучшение {improvement:.3f}, новая температура {new_temperature:.3f}")

    except Exception as e:
        print(f"Ошибка обработки RLHF корректировки: {e}")

def get_rlhf_params_for_category(category):
    if category in text_rlhf_policy:
        return text_rlhf_policy[category].get('best_params', {})
    return {}

image_rlhf_policy = {}

def load_image_rlhf_policy():
    global image_rlhf_policy
    try:
        image_rlhf_policy = db.load_image_rlhf_policy()
    except Exception as e:
        print(f"Ошибка загрузки image RLHF политики: {e}")
        image_rlhf_policy = {}

def save_image_rlhf_policy(category, data):
    try:
        db.save_image_rlhf_policy(category, data)
    except Exception as e:
        print(f"Ошибка сохранения image RLHF политики: {e}")

def image_reinforcement_learning_update(category, reward, generation_params):
    global image_rlhf_policy
    try:
        if category not in image_rlhf_policy:
            image_rlhf_policy[category] = {
                'rewards': [],
                'avg_reward': 0.0,
                'count': 0,
                'best_params': generation_params
            }

        image_rlhf_policy[category]['rewards'].append(reward)
        image_rlhf_policy[category]['count'] += 1
        image_rlhf_policy[category]['avg_reward'] = sum(image_rlhf_policy[category]['rewards']) / len(image_rlhf_policy[category]['rewards'])

        if reward > image_rlhf_policy[category]['avg_reward']:
            image_rlhf_policy[category]['best_params'] = generation_params.copy()

        if len(image_rlhf_policy[category]['rewards']) > 100:
            image_rlhf_policy[category]['rewards'] = image_rlhf_policy[category]['rewards'][-100:]

        save_image_rlhf_policy(category, image_rlhf_policy[category])
        print(f"RLHF изображений: обновлена политика для '{category}', средняя оценка: {image_rlhf_policy[category]['avg_reward']:.3f}")

    except Exception as e:
        print(f"Ошибка обновления image RLHF: {e}")

def get_image_rlhf_params_for_category(category):
    if category in image_rlhf_policy:
        return image_rlhf_policy[category].get('best_params', {})
    return {}


def download_real_images(query, num_images=3):
    if not os.path.exists('image_data'):
        os.makedirs('image_data')

    downloaded_images = []

    # Словарь для перевода русских слов на английские для поиска
    translate_dict = {
        'кот': 'cat', 'котик': 'cat', 'котенок': 'kitten', 'кошка': 'cat',
        'собака': 'dog', 'пес': 'dog', 'щенок': 'puppy', 'собачка': 'dog',
        'небо': 'sky', 'облака': 'clouds', 'тучи': 'clouds', 'облако': 'cloud',
        'море': 'ocean', 'вода': 'water', 'озеро': 'lake', 'река': 'river',
        'лес': 'forest', 'дерево': 'tree', 'деревья': 'trees', 'парк': 'park',
        'гора': 'mountain', 'горы': 'mountains', 'холм': 'hill',
        'цветок': 'flower', 'цветы': 'flowers', 'роза': 'rose',
        'солнце': 'sun', 'луна': 'moon', 'звезды': 'stars', 'небеса': 'sky',
        'дом': 'house', 'здание': 'building', 'квартира': 'apartment',
        'машина': 'car', 'автомобиль': 'car', 'транспорт': 'transport',
        'птица': 'bird', 'рыба': 'fish', 'животное': 'animal',
        'огонь': 'fire', 'пламя': 'flame', 'жар': 'heat',
        'снег': 'snow', 'зима': 'winter', 'мороз': 'frost',
        'лето': 'summer', 'весна': 'spring', 'осень': 'autumn',
        'радость': 'joy', 'печаль': 'sadness', 'грусть': 'sadness',
        'город': 'city', 'мост': 'bridge', 'улица': 'street',
        'пляж': 'beach', 'корабль': 'ship', 'лодка': 'boat',
        'туман': 'fog', 'дым': 'smoke', 'буря': 'storm',
        'ветер': 'wind', 'дождь': 'rain', 'буря': 'storm',
        'звук': 'sound', 'музыка': 'music', 'песня': 'song',
        'еда': 'food', 'фрукты': 'fruits', 'овощи': 'vegetables',
        'кофе': 'coffee', 'чай': 'tea', 'вино': 'wine',
        'спорт': 'sport', 'футбол': 'football', 'баскетбол': 'basketball',
        'игра': 'game', 'компьютер': 'computer', 'интернет': 'internet',
        'программа': 'program', 'код': 'code', 'сайт': 'website',
        'телефон': 'phone', 'телевизор': 'tv', 'радио': 'radio',
        'учёба': 'study', 'школа': 'school', 'университет': 'university',
        'работа': 'work', 'офис': 'office', 'праздник': 'holiday',
        'путешествие': 'travel', 'отпуск': 'vacation', 'отдых': 'rest',
        'любовь': 'love', 'дружба': 'friendship', 'семья': 'family',
        'солнце светит': 'sun shines', 'светит луна': 'moon shines',
        'горит огонь': 'fire burns', 'падает снег': 'snow falls',
        'летит птица': 'bird flies', 'плывет корабль': 'ship sails',
        'бежит собака': 'dog runs', 'растет дерево': 'tree grows'
    }

    words = query.lower().split()
    english_words = []
    for word in words:
        if word in translate_dict:
            english_words.append(translate_dict[word])
        else:
            # Пытаемся найти похожие слова в словаре
            found = False
            for ru_word, en_word in translate_dict.items():
                if word in ru_word or ru_word in word:
                    english_words.append(en_word)
                    found = True
                    break
            if not found:
                english_words.append(word)

    search_query = ' '.join(english_words) if english_words else query

    # Источники изображений с корректным кодированием параметров
    image_sources = [
        lambda q, i: f"https://source.unsplash.com/800x600/?{urllib.parse.quote(q.replace(' ', ','))}",
        lambda q, i: f"https://picsum.photos/800/600?random={i}&blur=2",
        lambda q, i: f"https://source.unsplash.com/featured/800x600/?{urllib.parse.quote(q.replace(' ', ','))},nature",
        lambda q, i: f"https://source.unsplash.com/collection/190727/800x600/?{urllib.parse.quote(q.replace(' ', ','))}"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"🔍 Начинаю загрузку {num_images} изображений для запроса: '{search_query}'")

    for i in range(num_images):
        downloaded = False
        attempts = 0
        max_attempts = 3

        while not downloaded and attempts < max_attempts:
            try:
                attempts += 1
                source_idx = (i + attempts - 1) % len(image_sources)
                url = image_sources[source_idx](search_query, i + attempts)

                print(f"📥 Попытка {attempts}/{max_attempts} для изображения {i+1}")
                print(f"🌐 Загружаю из: {url}")

                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 200 and len(response.content) > 1000:
                    file_hash = hashlib.md5(f"{query}_{search_query}_{i}_{attempts}".encode('utf-8')).hexdigest()
                    filename = f"image_data/real_{file_hash}.jpg"

                    with open(filename, 'wb') as f:
                        f.write(response.content)

                    # Проверяем валидность изображения
                    try:
                        with Image.open(filename) as img:
                            img.verify()
                        downloaded_images.append(filename)
                        downloaded = True
                        print(f"✅ Изображение {i+1}/{num_images} успешно скачано: {filename}")
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"❌ Скачанный файл не является валидным изображением: {e}")
                        if os.path.exists(filename):
                            os.remove(filename)
                else:
                    print(f"⚠️ Плохой ответ: status={response.status_code}, size={len(response.content)}")

            except requests.exceptions.Timeout:
                print(f"⏰ Таймаут при загрузке изображения {i+1}, попытка {attempts}")
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"🌐 Сетевая ошибка при загрузке изображения {i+1}: {e}")
                time.sleep(1)

            except Exception as e:
                print(f"❌ Ошибка при загрузке изображения {i+1}: {e}")
                time.sleep(1)

        if not downloaded:
            print(f"⚠️ Не удалось загрузить изображение {i+1} после {max_attempts} попыток")

    print(f"📊 Итого загружено: {len(downloaded_images)}/{num_images} изображений")
    return downloaded_images

def load_improved_params():
    """
    Загружает улучшенные параметры генерации из базы данных.
    Возвращает словарь с параметрами по категориям.
    """
    try:
        rows = db.select_from('improved_generation_params')
        params = {}
        for row in rows:
            params[row['category']] = json.loads(row['best_params'])
        return params
    except Exception as e:
        print(f"Ошибка загрузки улучшенных параметров из БД: {e}")
        return {}

def generate_smart_base_images(query, num_images=3):
    """Создает реалистичные изображения с улучшенной детализацией"""
    if not os.path.exists('image_data'):
        os.makedirs('image_data')

    # Загружаем улучшенные параметры
    improved_params = load_improved_params()

    # Улучшенные темы с более реалистичными настройками
    themes = {
        # Природа
        'небо': {'colors': [(87, 160, 211), (130, 200, 240), (255, 255, 255), (220, 240, 255)], 'type': 'sky'},
        'sky': {'colors': [(87, 160, 211), (130, 200, 240), (255, 255, 255), (220, 240, 255)], 'type': 'sky'},
        'море': {'colors': [(32, 78, 136), (64, 164, 223), (135, 206, 235), (0, 139, 204)], 'type': 'ocean'},
        'ocean': {'colors': [(32, 78, 136), (64, 164, 223), (135, 206, 235), (0, 139, 204)], 'type': 'ocean'},
        'лес': {'colors': [(34, 80, 34), (60, 120, 60), (85, 140, 85), (40, 60, 40)], 'type': 'forest'},
        'forest': {'colors': [(34, 80, 34), (60, 120, 60), (85, 140, 85), (40, 60, 40)], 'type': 'forest'},
        'солнце': {'colors': [(255, 200, 0), (255, 165, 0), (255, 140, 0), (255, 215, 0)], 'type': 'sun'},
        'sun': {'colors': [(255, 200, 0), (255, 165, 0), (255, 140, 0), (255, 215, 0)], 'type': 'sun'},

        # Животные
        'кот': {'colors': [(160, 110, 70), (200, 150, 100), (180, 130, 85), (140, 90, 50)], 'type': 'cat'},
        'cat': {'colors': [(160, 110, 70), (200, 150, 100), (180, 130, 85), (140, 90, 50)], 'type': 'cat'},
        'собака': {'colors': [(120, 80, 50), (160, 110, 70), (200, 150, 100), (100, 60, 30)], 'type': 'dog'},
        'dog': {'colors': [(120, 80, 50), (160, 110, 70), (200, 150, 100), (100, 60, 30)], 'type': 'dog'},

        # Огонь
        'огонь': {'colors': [(200, 40, 0), (255, 100, 0), (255, 150, 0), (255, 200, 50)], 'type': 'fire'},
        'fire': {'colors': [(200, 40, 0), (255, 100, 0), (255, 150, 0), (255, 200, 50)], 'type': 'fire'},

        # Космос
        'space': {'colors': [(10, 10, 40), (25, 25, 80), (50, 50, 120), (5, 5, 20)], 'type': 'space'},
        'космос': {'colors': [(10, 10, 40), (25, 25, 80), (50, 50, 120), (5, 5, 20)], 'type': 'space'},

        # Для неизвестных слов
        'default': {'colors': [(100, 150, 200), (150, 100, 200), (200, 100, 150), (120, 180, 160)], 'type': 'abstract'}
    }

    # Определяем тему
    query_lower = query.lower()
    theme = None
    for key in themes:
        if key in query_lower:
            theme = themes[key]
            break

    if not theme:
        # Для неизвестных слов пытаемся найти похожие
        try:
            if 'improve_unknown_word_generation' in locals():
                suggested_theme = improve_unknown_word_generation(query)
                if suggested_theme:
                    # Используем подходящую тему из расширенного списка
                    theme = themes.get(suggested_theme, themes['default'])
                    print(f"Определена категория '{suggested_theme}' для запроса '{query}'")

                    # Применяем улучшенные параметры если они есть
                    if suggested_theme in improved_params:
                        improved_data = improved_params[suggested_theme]
                        print(f"Применяем улучшенные параметры для '{suggested_theme}' (средняя оценка: {improved_data.get('avg_score', 0):.3f})")

                        # Можем модифицировать цвета на основе успешных примеров
                        if improved_data.get('avg_score', 0) > 0.7:
                            # Для очень успешных категорий используем более яркие цвета
                            if 'colors' in theme:
                                theme['colors'] = [
                                    tuple(min(255, int(c * 1.2)) for c in color) 
                                    for color in theme['colors']
                                ]
                else:
                    theme = themes['default']
            else:
                theme = themes['default']
        except Exception as e:
            print(f"Ошибка при определении темы: {e}")
            theme = themes['default']

    generated_images = []

    for i in range(num_images):
        try:
            # Создаем изображение высокого разрешения
            img = Image.new('RGB', (800, 800))
            draw = ImageDraw.Draw(img)

            # Создаем более сложный градиентный фон
            def create_gradient_background(img, colors):
                """Создает многослойный градиентный фон"""
                width, height = img.size
                
                # Основной градиент
                for y in range(height):
                    color_progress = y / height
                    
                    # Выбираем между цветами плавно
                    color_index = color_progress * (len(colors) - 1)
                    base_idx = int(color_index)
                    next_idx = min(base_idx + 1, len(colors) - 1)
                    blend_factor = color_index - base_idx
                    
                    base_color = colors[base_idx]
                    next_color = colors[next_idx]
                    
                    r = int(base_color[0] * (1 - blend_factor) + next_color[0] * blend_factor)
                    g = int(base_color[1] * (1 - blend_factor) + next_color[1] * blend_factor)
                    b = int(base_color[2] * (1 - blend_factor) + next_color[2] * blend_factor)
                    
                    # Добавляем небольшое случайное варьирование для реалистичности
                    r = max(0, min(255, r + random.randint(-5, 5)))
                    g = max(0, min(255, g + random.randint(-5, 5)))
                    b = max(0, min(255, b + random.randint(-5, 5)))
                    
                    draw.line([(0, y), (width, y)], fill=(r, g, b))

            create_gradient_background(img, theme['colors'])

            # Рисуем реалистичные элементы в зависимости от типа
            img_type = theme.get('type', 'abstract')

            def draw_realistic_clouds(draw, colors, width, height):
                """Рисует реалистичные облака"""
                for _ in range(random.randint(5, 12)):
                    # Основное положение облака
                    cx = random.randint(0, width)
                    cy = random.randint(0, height // 2)
                    
                    # Создаем облако из нескольких пересекающихся кругов
                    num_circles = random.randint(8, 15)
                    base_size = random.randint(40, 80)
                    
                    cloud_color = (255, 255, 255, random.randint(180, 220))
                    
                    for _ in range(num_circles):
                        offset_x = random.randint(-base_size, base_size)
                        offset_y = random.randint(-base_size//2, base_size//2)
                        circle_size = base_size + random.randint(-20, 20)
                        
                        x = cx + offset_x
                        y = cy + offset_y
                        
                        # Используем более мягкие цвета
                        alpha = random.randint(150, 200)
                        soft_color = (255, 255, 255)
                        
                        draw.ellipse([x, y, x + circle_size, y + circle_size//2], fill=soft_color)

            def draw_realistic_ocean(draw, colors, width, height):
                """Рисует реалистичные волны"""
                # Рисуем несколько слоев волн
                for layer in range(6):
                    y_base = height // 2 + layer * 60
                    wave_color = colors[layer % len(colors)]
                    
                    # Создаем синусоидальные волны
                    points = []
                    for x in range(0, width + 20, 8):
                        wave1 = 15 * math.sin(x * 0.02 + layer * 1.5)
                        wave2 = 8 * math.sin(x * 0.05 + layer * 2)
                        wave3 = 4 * math.sin(x * 0.1 + layer * 3)
                        
                        y = y_base + wave1 + wave2 + wave3
                        points.extend([x, y])
                    
                    # Добавляем нижний край для заливки
                    points.extend([width, height, 0, height])
                    
                    if len(points) >= 6:
                        draw.polygon(points, fill=wave_color)

            def draw_realistic_forest(draw, colors, width, height):
                """Рисует реалистичный лес"""
                # Рисуем деревья разных размеров
                for _ in range(random.randint(8, 15)):
                    x = random.randint(0, width)
                    tree_height = random.randint(100, 200)
                    tree_width = random.randint(20, 40)
                    
                    # Ствол
                    trunk_color = (101, 67, 33)
                    trunk_width = tree_width // 4
                    draw.rectangle([
                        x - trunk_width//2, height - 30,
                        x + trunk_width//2, height - tree_height//3
                    ], fill=trunk_color)
                    
                    # Крона - несколько слоев зеленого
                    crown_layers = 3
                    for layer in range(crown_layers):
                        crown_size = tree_width - layer * 5
                        crown_y = height - tree_height//3 - layer * 20
                        crown_color = colors[layer % len(colors)]
                        
                        # Рисуем крону как неправильный круг
                        for _ in range(6):
                            offset_x = random.randint(-crown_size//3, crown_size//3)
                            offset_y = random.randint(-crown_size//4, crown_size//4)
                            circle_size = crown_size + random.randint(-5, 5)
                            
                            draw.ellipse([
                                x + offset_x - circle_size//2,
                                crown_y + offset_y - circle_size//2,
                                x + offset_x + circle_size//2,
                                crown_y + offset_y + circle_size//2
                            ], fill=crown_color)

            def draw_realistic_fire(draw, colors, width, height):
                """Рисует реалистичное пламя"""
                # Основание огня
                base_y = height - 50
                base_width = width // 3
                base_x = width // 2 - base_width // 2
                
                # Рисуем множество языков пламени
                for flame in range(12):
                    flame_x = base_x + random.randint(0, base_width)
                    flame_height = random.randint(80, 180)
                    flame_width = random.randint(15, 35)
                    
                    # Создаем изгибающийся язык пламени
                    points = []
                    segments = 8
                    
                    for seg in range(segments + 1):
                        y_pos = base_y - (seg / segments) * flame_height
                        
                        # Добавляем изгиб
                        curve_offset = 15 * math.sin(seg * 0.8) * (seg / segments)
                        x_pos = flame_x + curve_offset
                        
                        # Ширина уменьшается к верху
                        width_factor = 1 - (seg / segments) * 0.7
                        current_width = flame_width * width_factor
                        
                        # Левая и правая стороны пламени
                        if seg == 0:
                            points.extend([x_pos - current_width//2, y_pos])
                        points.extend([x_pos + current_width//2, y_pos])
                    
                    # Добавляем правую сторону в обратном порядке
                    for seg in range(segments, -1, -1):
                        y_pos = base_y - (seg / segments) * flame_height
                        curve_offset = 15 * math.sin(seg * 0.8) * (seg / segments)
                        x_pos = flame_x + curve_offset
                        width_factor = 1 - (seg / segments) * 0.7
                        current_width = flame_width * width_factor
                        
                        points.extend([x_pos - current_width//2, y_pos])
                    
                    if len(points) >= 6:
                        flame_color = colors[flame % len(colors)]
                        draw.polygon(points, fill=flame_color)

            # Применяем соответствующий метод рисования
            if img_type == 'sky':
                draw_realistic_clouds(draw, theme['colors'], img.width, img.height)
            elif img_type == 'ocean':
                draw_realistic_ocean(draw, theme['colors'], img.width, img.height)
            elif img_type == 'forest':
                draw_realistic_forest(draw, theme['colors'], img.width, img.height)
            elif img_type == 'fire':
                draw_realistic_fire(draw, theme['colors'], img.width, img.height)

            elif img_type == 'cat':
                def draw_realistic_cat(draw, colors, width, height):
                    """Рисует более реалистичного кота"""
                    # Позиция кота
                    cat_x = width // 2 - 100
                    cat_y = height // 2 - 50
                    
                    main_color = colors[0]
                    accent_color = colors[1] if len(colors) > 1 else main_color
                    
                    # Тело кота (овальное)
                    body_width, body_height = 120, 80
                    draw.ellipse([
                        cat_x, cat_y + 40,
                        cat_x + body_width, cat_y + 40 + body_height
                    ], fill=main_color)
                    
                    # Голова (круглая)
                    head_size = 70
                    head_x = cat_x + 25
                    head_y = cat_y
                    draw.ellipse([
                        head_x, head_y,
                        head_x + head_size, head_y + head_size
                    ], fill=main_color)
                    
                    # Уши (треугольники)
                    ear_size = 25
                    # Левое ухо
                    draw.polygon([
                        (head_x + 10, head_y + 15),
                        (head_x + 10 + ear_size, head_y),
                        (head_x + 20 + ear_size, head_y + 20)
                    ], fill=main_color)
                    # Правое ухо
                    draw.polygon([
                        (head_x + 30, head_y + 15),
                        (head_x + 30 + ear_size, head_y),
                        (head_x + 40 + ear_size, head_y + 20)
                    ], fill=main_color)
                    
                    # Глаза (зеленые овалы)
                    eye_color = (34, 139, 34)
                    draw.ellipse([head_x + 15, head_y + 25, head_x + 25, head_y + 35], fill=eye_color)
                    draw.ellipse([head_x + 40, head_y + 25, head_x + 50, head_y + 35], fill=eye_color)
                    
                    # Зрачки
                    draw.ellipse([head_x + 18, head_y + 28, head_x + 22, head_y + 32], fill=(0, 0, 0))
                    draw.ellipse([head_x + 43, head_y + 28, head_x + 47, head_y + 32], fill=(0, 0, 0))
                    
                    # Нос (розовый треугольник)
                    nose_color = (255, 182, 193)
                    draw.polygon([
                        (head_x + 32, head_y + 40),
                        (head_x + 38, head_y + 40),
                        (head_x + 35, head_y + 45)
                    ], fill=nose_color)
                    
                    # Усы
                    whisker_color = (64, 64, 64)
                    for i in range(3):
                        # Левые усы
                        y_pos = head_y + 45 + i * 3
                        draw.line([head_x - 10, y_pos, head_x + 20, y_pos], fill=whisker_color, width=1)
                        # Правые усы
                        draw.line([head_x + 50, y_pos, head_x + 80, y_pos], fill=whisker_color, width=1)
                    
                    # Хвост (изогнутый)
                    tail_points = []
                    tail_start_x = cat_x + body_width - 10
                    tail_start_y = cat_y + 60
                    
                    for i in range(15):
                        angle = i * 0.3
                        x = tail_start_x + i * 8 + 20 * math.sin(angle)
                        y = tail_start_y - i * 4 + 10 * math.cos(angle)
                        tail_points.extend([x, y])
                    
                    if len(tail_points) >= 4:
                        for i in range(0, len(tail_points) - 2, 4):
                            if i + 3 < len(tail_points):
                                draw.line([tail_points[i], tail_points[i+1], tail_points[i+2], tail_points[i+3]], 
                                         fill=accent_color, width=8)

                draw_realistic_cat(draw, theme['colors'], img.width, img.height)

            elif img_type == 'sun':
                def draw_realistic_sun(draw, colors, width, height):
                    """Рисует реалистичное солнце"""
                    sun_x = width // 2
                    sun_y = height // 3
                    sun_radius = 80
                    
                    # Основное солнце (градиентный круг)
                    for r in range(sun_radius, 0, -3):
                        intensity = 1 - (r / sun_radius)
                        color_idx = int(intensity * (len(colors) - 1))
                        color = colors[min(color_idx, len(colors) - 1)]
                        
                        draw.ellipse([
                            sun_x - r, sun_y - r,
                            sun_x + r, sun_y + r
                        ], fill=color)
                    
                    # Лучи солнца
                    ray_color = colors[-1]
                    for angle in range(0, 360, 30):
                        rad = math.radians(angle)
                        start_x = sun_x + (sun_radius + 10) * math.cos(rad)
                        start_y = sun_y + (sun_radius + 10) * math.sin(rad)
                        end_x = sun_x + (sun_radius + 40) * math.cos(rad)
                        end_y = sun_y + (sun_radius + 40) * math.sin(rad)
                        
                        draw.line([start_x, start_y, end_x, end_y], fill=ray_color, width=4)

                draw_realistic_sun(draw, theme['colors'], img.width, img.height)

            elif img_type == 'space':
                def draw_realistic_space(draw, colors, width, height):
                    """Рисует реалистичный космос"""
                    # Звезды разных размеров и яркости
                    for _ in range(150):
                        x = random.randint(0, width)
                        y = random.randint(0, height)
                        
                        if random.random() > 0.95:  # Яркие звезды
                            size = random.randint(3, 6)
                            brightness = random.randint(200, 255)
                            star_color = (brightness, brightness, brightness)
                            
                            # Крестообразная звезда
                            draw.line([x-size, y, x+size, y], fill=star_color, width=1)
                            draw.line([x, y-size, x, y+size], fill=star_color, width=1)
                            draw.ellipse([x-1, y-1, x+1, y+1], fill=star_color)
                        else:  # Обычные звезды
                            size = random.randint(1, 3)
                            brightness = random.randint(150, 220)
                            star_color = (brightness, brightness, brightness)
                            draw.ellipse([x, y, x+size, y+size], fill=star_color)
                    
                    # Туманности
                    for _ in range(3):
                        nebula_x = random.randint(100, width-100)
                        nebula_y = random.randint(100, height-100)
                        nebula_size = random.randint(60, 120)
                        
                        nebula_color = colors[random.randint(1, len(colors)-1)]
                        
                        # Создаем туманность из нескольких прозрачных кругов
                        for _ in range(8):
                            offset_x = random.randint(-nebula_size//2, nebula_size//2)
                            offset_y = random.randint(-nebula_size//2, nebula_size//2)
                            circle_size = random.randint(nebula_size//3, nebula_size)
                            
                            # Делаем цвет более прозрачным
                            faded_color = tuple(int(c * 0.3) for c in nebula_color)
                            
                            draw.ellipse([
                                nebula_x + offset_x - circle_size//2,
                                nebula_y + offset_y - circle_size//2,
                                nebula_x + offset_x + circle_size//2,
                                nebula_y + offset_y + circle_size//2
                            ], fill=faded_color)

                draw_realistic_space(draw, theme['colors'], img.width, img.height)

            else:  # abstract и другие типы
                def draw_realistic_abstract(draw, colors, width, height):
                    """Рисует красивую абстракцию"""
                    # Создаем плавные волнообразные формы
                    for layer in range(5):
                        points = []
                        layer_color = colors[layer % len(colors)]
                        
                        # Создаем плавную кривую
                        for x in range(0, width + 50, 20):
                            wave1 = 30 * math.sin(x * 0.01 + layer)
                            wave2 = 20 * math.sin(x * 0.02 + layer * 2)
                            wave3 = 10 * math.sin(x * 0.05 + layer * 3)
                            
                            y = height // 2 + wave1 + wave2 + wave3 + layer * 40
                            points.extend([x, y])
                        
                        # Добавляем края для заливки
                        points.extend([width, height, 0, height])
                        
                        if len(points) >= 6:
                            draw.polygon(points, fill=layer_color)
                    
                    # Добавляем геометрические элементы
                    for _ in range(8):
                        x = random.randint(50, width - 50)
                        y = random.randint(50, height - 50)
                        size = random.randint(20, 60)
                        color = colors[random.randint(0, len(colors) - 1)]
                        
                        if random.random() > 0.5:
                            # Круги с мягкими краями
                            for r in range(size, 0, -5):
                                alpha = 1 - (r / size)
                                soft_color = tuple(int(c * alpha + 255 * (1 - alpha)) for c in color)
                                draw.ellipse([x - r//2, y - r//2, x + r//2, y + r//2], fill=soft_color)
                        else:
                            # Многоугольники
                            sides = random.randint(3, 8)
                            poly_points = []
                            for angle in range(0, 360, 360 // sides):
                                rad = math.radians(angle)
                                px = x + size * math.cos(rad)
                                py = y + size * math.sin(rad)
                                poly_points.extend([px, py])
                            
                            if len(poly_points) >= 6:
                                draw.polygon(poly_points, fill=color)

                draw_realistic_abstract(draw, theme['colors'], img.width, img.height)

            # Применяем профессиональные фильтры
            # Увеличиваем контраст
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Немного увеличиваем насыщенность
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.1)
            
            # Применяем мягкое размытие для сглаживания
            img = img.filter(ImageFilter.SMOOTH_MORE)
            
            # Уменьшаем до финального размера с антиалиасингом
            img = img.resize((512, 512), Image.Resampling.LANCZOS)

            # Сохраняем изображение
            filename = f"image_data/smart_{hashlib.md5(f'{query}_{i}_{time.time()}'.encode('utf-8')).hexdigest()}.jpg"
            img.save(filename, quality=90)
            generated_images.append(filename)
            
            print(f"✅ Создано реалистичное изображение {i+1}/{num_images}: {filename}")

        except Exception as e:
            print(f"Ошибка генерации изображения {i+1}: {e}")

    return generated_images

def analyze_image_colors(image_path):
    """Анализирует цветовую палитру изображения"""
    try:
        with Image.open(image_path) as img:
            # Уменьшаем размер для быстрого анализа
            img = img.resize((100, 100))
            img = img.convert('RGB')

            # Получаем все пиксели
            pixels = list(img.getdata())

            # Анализируем доминирующие цвета
            color_counts = {}
            for pixel in pixels:
                color_counts[pixel] = color_counts.get(pixel, 0) + 1

            # Топ-5 цветов
            top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            # Анализируем яркость и контраст
            avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)

            return {
                'dominant_colors': [color[0] for color in top_colors],
                'brightness': avg_brightness,
                'contrast': max(pixels)[0] - min(pixels)[0] if pixels else 0
            }
    except Exception as e:
        print(f"Ошибка анализа изображения {image_path}: {e}")
        return None

def generate_from_dataset(downloaded_images, query):
    """Генерирует изображение на основе скачанного датасета"""
    if not downloaded_images:
        return None

    try:
        # Анализируем все скачанные изображения
        color_palette = []
        brightness_values = []

        for img_path in downloaded_images:
            analysis = analyze_image_colors(img_path)
            if analysis:
                color_palette.extend(analysis['dominant_colors'])
                brightness_values.append(analysis['brightness'])

        if not color_palette:
            return None

        # Создаем новое изображение на основе анализа
        img = Image.new('RGB', (512, 512))
        draw = ImageDraw.Draw(img)

        # Используем анализированные цвета для градиента
        avg_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 128

        # Создаем градиент из доминирующих цветов
        for y in range(512):
            color_index = int((y / 512) * (len(color_palette) - 1))
            if color_index < len(color_palette):
                color = color_palette[color_index]

                # Корректируем яркость
                adjusted_color = tuple(
                    min(255, max(0, int(c * (avg_brightness / 128))))
                    for c in color
                )

                draw.line([(0, y), (512, y)], fill=adjusted_color)

        # Добавляем абстрактные формы на основе анализа
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, 400)
            y = random.randint(0, 400)
            size = random.randint(30, 100)

            # Выбираем случайный цвет из палитры
            color = random.choice(color_palette)

            # Рисуем различные формы
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])

            if shape_type == 'circle':
                draw.ellipse([x, y, x+size, y+size], fill=color)
            elif shape_type == 'rectangle':
                draw.rectangle([x, y, x+size, y+size], fill=color)
            elif shape_type == 'triangle':
                draw.polygon([
                    (x, y+size),
                    (x+size//2, y),
                    (x+size, y+size)
                ], fill=color)

        # Применяем фильтры для лучшего результата
        if random.random() > 0.5:
            img = img.filter(ImageFilter.SMOOTH)

        # Сохраняем результат
        filename = f"image_data/dataset_generated_{hashlib.md5(f'{query}_{time.time()}'.encode('utf-8')).hexdigest()}.jpg"
        img.save(filename)

        return filename

    except Exception as e:
        print(f"Ошибка генерации из датасета: {e}")
        return None

def intelligent_image_generation(prompt, bpe, max_images=5):
    """Интеллектуальная генерация изображений по вашей идее"""
    print(f"🧠 Запуск интеллектуальной генерации для: '{prompt}'")

    # 1. Анализируем токены
    tokens = bpe.encode(prompt.lower())
    words = prompt.lower().split()

    print(f"📝 Токены: {tokens}")
    print(f"🔤 Слова: {words}")

    # 2. Формируем поисковые запросы
    search_queries = []

    # Основной запрос
    search_queries.append(prompt)

    # Отдельные слова
    for word in words:
        if len(word) > 2:  # Игнорируем короткие слова
            search_queries.append(word)

    # Комбинации слов
    if len(words) > 1:
        for i in range(len(words)-1):
            search_queries.append(f"{words[i]} {words[i+1]}")

    # 3. Скачиваем изображения для каждого запроса
    all_downloaded_images = []

    for query in search_queries[:2]:  # Уменьшили количество запросов для стабильности
        print(f"🔍 Поиск изображений для: '{query}'")
        
        # Увеличиваем количество изображений за запрос
        downloaded = download_real_images(query, num_images=3)
        all_downloaded_images.extend(downloaded)
        
        # Добавляем паузу между запросами
        if downloaded:
            print(f"⏸️ Пауза 2 сек между запросами...")
            time.sleep(2)

        if len(all_downloaded_images) >= max_images:
            break

    print(f"📥 Итого скачано {len(all_downloaded_images)} изображений для анализа")

    # 4. Генерируем изображение на основе скачанного датасета
    generated_image = None

    if all_downloaded_images:
        print("🎨 Генерация изображения на основе скачанного датасета...")
        generated_image = generate_from_dataset(all_downloaded_images, prompt)

    # 5. Если не получилось сгенерировать из датасета, используем умную генерацию
    if not generated_image:
        print("🎭 Fallback: использую умную генерацию...")
        smart_images = generate_smart_base_images(prompt, num_images=1)
        generated_image = smart_images[0] if smart_images else None

    # 6. Сохраняем данные для обучения
    generation_data = {
        'prompt': prompt,
        'tokens': tokens,
        'words': words,
        'search_queries': search_queries,
        'downloaded_images': all_downloaded_images,
        'generated_image': generated_image,
        'timestamp': time.time(),
        'method': 'intelligent_dataset' if all_downloaded_images else 'smart_fallback'
    }

    # Сохраняем в файл для дальнейшего обучения
    data_file = 'intelligent_generation_data.json'
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    data.append(generation_data)

    # Ограничиваем размер файла
    if len(data) > 100:
        data = data[-100:]

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Данные сохранены в {data_file}")

    return generated_image

def generate_image_from_text(prompt, bpe, max_images=5):
    """Главная функция генерации изображений с улучшенной реалистичностью"""
    try:
        print(f"🎨 Генерация изображения для: '{prompt}'")
        start_time = time.time()

        # Приоритет: сначала пробуем интеллектуальную систему с реальными изображениями
        generated_image = None
        
        # Шаг 1: Пробуем загрузить реальные изображения и генерировать на их основе
        print("🌐 Попытка загрузки реальных изображений...")
        real_images = download_real_images(prompt, num_images=3)
        
        if real_images:
            print(f"✅ Загружено {len(real_images)} реальных изображений")
            generated_image = generate_from_dataset(real_images, prompt)
            if generated_image:
                print("✅ Создано изображение на основе реальных данных")
        
        # Шаг 2: Если не получилось с реальными, используем интеллектуальную систему
        if not generated_image:
            print("🧠 Использую интеллектуальную генерацию...")
            generated_image = intelligent_image_generation(prompt, bpe, max_images)
        
        # Шаг 3: Fallback на улучшенную умную генерацию
        if not generated_image:
            print("🎭 Fallback: умная генерация базовых изображений...")
            smart_images = generate_smart_base_images(prompt, num_images=1)
            generated_image = smart_images[0] if smart_images else None

        generation_time = time.time() - start_time
        print(f"⏱️ Время генерации: {generation_time:.2f} секунд")

        if generated_image:
            print(f"✅ Изображение сгенерировано: {generated_image}")
            
            # Проверяем что файл действительно существует
            if os.path.exists(generated_image):
                file_size = os.path.getsize(generated_image)
                print(f"📁 Размер файла: {file_size} байт")
                
                # Применяем RLHF для улучшения качества
                try:
                    category = improve_unknown_word_generation(prompt) or 'default'
                    quality = evaluate_image_quality(generated_image)
                    auto_score = quality.get('auto_score', 0.5)
                    
                    # Если автоматическая оценка высокая, сохраняем параметры
                    if auto_score > 0.7:
                        generation_params = {
                            'method': 'dataset' if real_images else 'intelligent',
                            'real_images_count': len(real_images) if real_images else 0,
                            'prompt': prompt,
                            'auto_score': auto_score
                        }
                        reinforcement_learning_update(category, auto_score, generation_params)
                        print(f"🧠 RLHF: Сохранены успешные параметры для '{category}' (оценка: {auto_score:.3f})")
                except Exception as e:
                    print(f"⚠️ Ошибка RLHF: {e}")
                
                return generated_image
            else:
                print(f"❌ Файл не найден: {generated_image}")
                return None
        else:
            print("❌ Не удалось сгенерировать изображение")
            return None

    except Exception as e:
        print(f"❌ Ошибка генерации изображения: {e}")
        import traceback
        traceback.print_exc()
        return None

def improve_unknown_word_generation(query):
    """Улучшает определение категории для неизвестных слов"""
    translate_dict = {
        'небо': 'sky',
        'море': 'ocean',
        'лес': 'forest',
        'солнце': 'sun',
        'кот': 'cat',
        'собака': 'dog',
        'огонь': 'fire',
        'космос': 'space',
        'здание': 'building',
        'машина': 'transport',
        'еда': 'food',
    }
    for ru_word, en_word in translate_dict.items():
        if ru_word in query.lower():
            return en_word
    return None

# Функции для RLHF
def evaluate_image_quality(image_path, user_rating=None):
    """Оценивает качество изображения"""
    try:
        # Простая автоматическая оценка на основе контраста и яркости
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            pixels = list(img_gray.getdata())
            
            # Контраст
            contrast = (max(pixels) - min(pixels)) / 255.0 if pixels else 0
            
            # Яркость
            brightness = sum(pixels) / (len(pixels) * 255.0) if pixels else 0
            
            # Комбинированная оценка
            auto_score = (contrast * 0.3 + brightness * 0.7)
            
            return {
                'auto_score': auto_score,
                'user_rating': user_rating,
                'contrast': contrast,
                'brightness': brightness
            }
    except Exception as e:
        print(f"Ошибка оценки изображения: {e}")
        return {'auto_score': 0.5, 'user_rating': user_rating}

def reinforcement_learning_update(category, reward, generation_params):
    """
    Обновляет параметры генерации на основе RLHF в базе данных.
    """
    try:
        # Загружаем текущие параметры из базы
        rows = db.select_from('improved_generation_params', where='category = ?', params=(category,))
        import json
        if rows:
            data = rows[0]
            scores = json.loads(data['scores'])
            scores.append(reward)
            if len(scores) > 50:
                scores = scores[-50:]
            avg_score = sum(scores) / len(scores)
            best_params = json.loads(data['best_params'])
            if reward > avg_score:
                best_params = generation_params.copy()
            # Обновляем запись
            db.update('improved_generation_params',
                      where='category = ?',
                      where_params=(category,),
                      scores=json.dumps(scores, ensure_ascii=False),
                      avg_score=avg_score,
                      best_params=json.dumps(best_params, ensure_ascii=False),
                      total_generations=data['total_generations'] + 1)
        else:
            # Вставляем новую запись
            db.insert_into('improved_generation_params',
                           category=category,
                           scores=json.dumps([reward], ensure_ascii=False),
                           avg_score=reward,
                           best_params=json.dumps(generation_params, ensure_ascii=False),
                           total_generations=1)
        print(f"RLHF: Обновлены параметры для '{category}' с новой оценкой {reward:.3f}")
    except Exception as e:
        print(f"Ошибка обновления RLHF в БД: {e}")

load_text_rlhf_policy()
load_image_rlhf_policy()