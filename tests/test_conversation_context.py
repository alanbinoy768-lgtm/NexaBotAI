from src.conversation_manager import contextual_hint, is_follow_up


def test_follow_up_detection():
    assert is_follow_up("Who created it?", [{"text": "What is Python?"}]) is True


def test_contextual_hint_for_python_question():
    hint = contextual_hint("Who created it?", [{"text": "What is Python?"}])
    assert "Guido van Rossum" in hint
