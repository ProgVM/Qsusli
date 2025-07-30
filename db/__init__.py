import sqlite3
import threading
import time
import json
from sqlite3 import Row

class DB:
    _lock = threading.Lock()

    def __init__(self, path: str = 'main.db'):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = Row
        self.cur = self.conn.cursor()
        self.initialize_tables()

    def _execute(self, sql: str, params=()):
        with self._lock:
            self.cur.execute(sql, params)
            self.conn.commit()

    def select_from(self, table, columns='*', where=None, params=()):
        sql = f"SELECT {columns} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        with self._lock:
            self.cur.execute(sql, params)
            return self.cur.fetchall()

    def insert_into(self, table, **kwargs):
        keys = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        values = tuple(kwargs.values())
        sql = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        with self._lock:
            self.cur.execute(sql, values)
            self.conn.commit()

    def update(self, table, where, where_params=(), **kwargs):
        set_clause = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        values = tuple(kwargs.values())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        with self._lock:
            self.cur.execute(sql, values + where_params)
            self.conn.commit()

    def initialize_tables(self):
        """Создаёт все необходимые таблицы для ИИ."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS group_settings (
                chat_id TEXT NOT NULL,
                setting_key TEXT NOT NULL,
                setting_value TEXT,
                PRIMARY KEY (chat_id, setting_key)
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS train_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS image_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS text_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                rating INTEGER NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS image_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                image_path TEXT NOT NULL,
                rating INTEGER NOT NULL,
                normalized_rating REAL,
                auto_score REAL,
                timestamp REAL NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS text_rlhf_policy (
                category TEXT PRIMARY KEY,
                rewards TEXT NOT NULL,
                avg_reward REAL NOT NULL,
                count INTEGER NOT NULL,
                best_params TEXT NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS image_rlhf_policy (
                category TEXT PRIMARY KEY,
                rewards TEXT NOT NULL,
                avg_reward REAL NOT NULL,
                count INTEGER NOT NULL,
                best_params TEXT NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                original_response TEXT NOT NULL,
                corrected_response TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS improved_generation_params (
                category TEXT PRIMARY KEY,
                scores TEXT,
                avg_score REAL,
                best_params TEXT,
                total_generations INTEGER
            )
        """)
        self.conn.commit()

    # Методы работы с настройками групп
    def get_group_setting(self, chat_id, key):
        with self._lock:
            self.cur.execute("""
                SELECT setting_value FROM group_settings
                WHERE chat_id = ? AND setting_key = ?
            """, (str(chat_id), key))
            row = self.cur.fetchone()
            return row['setting_value'] if row else None
    
    def set_group_setting(self, chat_id, key, value):
        with self._lock:
            self.cur.execute("""
                INSERT INTO group_settings (chat_id, setting_key, setting_value)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id, setting_key) DO UPDATE SET setting_value=excluded.setting_value
            """, (str(chat_id), key, str(value)))
            self.conn.commit()
    
    def load_all_group_settings(self):
        with self._lock:
            self.cur.execute("SELECT chat_id, setting_key, setting_value FROM group_settings")
            rows = self.cur.fetchall()
            settings = {}
            for row in rows:
                chat_id = row['chat_id']
                key = row['setting_key']
                value = row['setting_value']
                if chat_id not in settings:
                    settings[chat_id] = {}
                settings[chat_id][key] = value
            return settings
    
    # Методы для тренировочных данных
    def load_train_data(self, limit=1000):
        with self._lock:
            self.cur.execute("SELECT input_text, output_text FROM train_data ORDER BY timestamp DESC LIMIT ?", (limit,))
            return self.cur.fetchall()

    def save_train_pair(self, input_text, output_text):
        self._execute(
            "INSERT INTO train_data (input_text, output_text, timestamp) VALUES (?, ?, ?)",
            (input_text, output_text, time.time())
        )

    def clear_train_data(self):
        self._execute("DELETE FROM train_data")

    # Методы для рейтингов
    def insert_text_rating(self, user_id, query, response, rating):
        self._execute(
            "INSERT INTO text_ratings (user_id, query, response, rating, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, response, rating, time.time())
        )

    def insert_image_rating(self, user_id, query, image_path, rating):
        self._execute(
            "INSERT INTO image_ratings (user_id, query, image_path, rating, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, image_path, rating, time.time())
        )

    # Методы для RLHF политики текста
    def load_text_rlhf_policy(self):
        rows = self.select_from("text_rlhf_policy")
        policy = {}
        for row in rows:
            policy[row['category']] = {
                'rewards': json.loads(row['rewards']),
                'avg_reward': row['avg_reward'],
                'count': row['count'],
                'best_params': json.loads(row['best_params'])
            }
        return policy

    def save_text_rlhf_policy(self, category, data):
        rewards_json = json.dumps(data['rewards'], ensure_ascii=False)
        best_params_json = json.dumps(data['best_params'], ensure_ascii=False)
        existing = self.select_from("text_rlhf_policy", where="category = ?", params=(category,))
        if existing:
            self.update("text_rlhf_policy",
                        where="category = ?",
                        where_params=(category,),
                        rewards=rewards_json,
                        avg_reward=data['avg_reward'],
                        count=data['count'],
                        best_params=best_params_json)
        else:
            self.insert_into("text_rlhf_policy",
                             category=category,
                             rewards=rewards_json,
                             avg_reward=data['avg_reward'],
                             count=data['count'],
                             best_params=best_params_json)

    # Методы для RLHF политики изображений
    def load_image_rlhf_policy(self):
        rows = self.select_from("image_rlhf_policy")
        policy = {}
        for row in rows:
            policy[row['category']] = {
                'rewards': json.loads(row['rewards']),
                'avg_reward': row['avg_reward'],
                'count': row['count'],
                'best_params': json.loads(row['best_params'])
            }
        return policy

    def save_image_rlhf_policy(self, category, data):
        rewards_json = json.dumps(data['rewards'], ensure_ascii=False)
        best_params_json = json.dumps(data['best_params'], ensure_ascii=False)
        existing = self.select_from("image_rlhf_policy", where="category = ?", params=(category,))
        if existing:
            self.update("image_rlhf_policy",
                        where="category = ?",
                        where_params=(category,),
                        rewards=rewards_json,
                        avg_reward=data['avg_reward'],
                        count=data['count'],
                        best_params=best_params_json)
        else:
            self.insert_into("image_rlhf_policy",
                             category=category,
                             rewards=rewards_json,
                             avg_reward=data['avg_reward'],
                             count=data['count'],
                             best_params=best_params_json)

    def save_correction(self, user_id, query, original_response, corrected_response, timestamp):
        self._execute(
            "INSERT INTO corrections (user_id, query, original_response, corrected_response, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, original_response, corrected_response, timestamp)
        )

    def load_corrections(self, limit=100):
        with self._lock:
            self.cur.execute("SELECT * FROM corrections ORDER BY timestamp DESC LIMIT ?", (limit,))
            return self.cur.fetchall()

    def close(self):
        self.conn.close()