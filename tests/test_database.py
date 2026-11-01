from src.database import ChatDatabase


def test_database_round_trip(tmp_path):
    db = ChatDatabase(tmp_path / "test_chat.db")
    cid = db.create_conversation("Test")
    mid = db.save_message(cid, "user", "Hello", model_used="qwen2.5:1.5b")
    messages = db.list_messages(cid)
    assert messages[0]["message_text"] == "Hello"
    assert messages[0]["id"] == mid
    assert messages[0]["model_used"] == "qwen2.5:1.5b"


def test_existing_database_is_migrated_without_losing_messages(tmp_path):
    import sqlite3

    path = tmp_path / "legacy_chat.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, sender TEXT NOT NULL, message_text TEXT NOT NULL, timestamp TEXT NOT NULL, predicted_intent TEXT, confidence REAL)"
        )
        connection.execute(
            "INSERT INTO conversations VALUES (1, 'Old chat', '2026-01-01', '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO messages VALUES (1, 1, 'user', 'Keep this', '2026-01-01', NULL, NULL)"
        )

    db = ChatDatabase(path)
    messages = db.list_messages(1)

    assert len(messages) == 1
    assert messages[0]["message_text"] == "Keep this"
    assert messages[0]["model_used"] is None


def test_clear_conversation_removes_messages_but_keeps_conversation(tmp_path):
    db = ChatDatabase(tmp_path / "clear_chat.db")
    conversation_id = db.create_conversation("Keep")
    db.save_message(conversation_id, "user", "Remove me")

    db.clear_conversation(conversation_id)

    assert db.get_conversation(conversation_id)["title"] == "Keep"
    assert db.list_messages(conversation_id) == []
