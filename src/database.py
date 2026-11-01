import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import DB_PATH, ensure_directories


class ChatDatabase:
    def __init__(self, db_path: str | Path = DB_PATH):
        ensure_directories()
        self.db_path = str(db_path)
        self.init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    sender TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    predicted_intent TEXT,
                    confidence REAL,
                    model_used TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            message_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "model_used" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN model_used TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    helpful INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def create_conversation(self, title: str) -> int:
        import datetime
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, timestamp, timestamp),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def list_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, sender, message_text, timestamp, predicted_intent, confidence, model_used FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC, id ASC",
                (conversation_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def save_message(
        self,
        conversation_id: int,
        sender: str,
        message_text: str,
        predicted_intent: Optional[str] = None,
        confidence: Optional[float] = None,
        model_used: Optional[str] = None,
    ) -> int:
        import datetime
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, sender, message_text, timestamp, predicted_intent, confidence, model_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, sender, message_text, timestamp, predicted_intent, confidence, model_used),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def delete_conversation(self, conversation_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

    def clear_conversation(self, conversation_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM feedback WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.commit()

    def clear_all(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
            conn.execute("DELETE FROM feedback")
            conn.commit()

    def save_feedback(self, conversation_id: int, message_id: int, helpful: bool):
        import datetime
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (conversation_id, message_id, helpful, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, message_id, int(helpful), timestamp),
            )
            conn.commit()
