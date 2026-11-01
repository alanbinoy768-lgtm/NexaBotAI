import pytest

import src.nlp_engine as nlp_engine
from src.nlp_engine import load_intent_dataset, predict_intent, train_intent_model
from src.response_generator import generate_response


def test_intent_dataset_loads():
    data = load_intent_dataset()
    assert "intents" in data
    assert len(data["intents"]) >= 20


def test_model_training_and_prediction():
    metrics = train_intent_model()
    assert metrics["accuracy"] >= 0
    result = predict_intent("hello there")
    assert result["intent"] == "greeting"
    assert 0 <= result["confidence"] <= 1


def test_required_messages_have_expected_intents():
    train_intent_model()
    expected_intents = {
        "Hello": "greeting",
        "Hello there!": "greeting",
        "Hi": "greeting",
        "Hey": "greeting",
        "Hey there": "greeting",
        "Good morning": "greeting",
        "Good afternoon": "greeting",
        "Good evening": "greeting",
        "How are you?": "greeting",
        "What's up?": "greeting",
        "Hi NexaBot": "greeting",
        "Hello chatbot": "greeting",
        "What is Python?": "programming",
        "Explain artificial intelligence.": "technology",
        "What is machine learning?": "technology",
        "Tell me a joke.": "jokes",
        "Who are you?": "identity",
        "What can you do?": "chatbot_capabilities",
        "Thank you.": "thanks",
        "Goodbye.": "goodbye",
    }

    for message, expected_intent in expected_intents.items():
        prediction = predict_intent(message)
        assert prediction["intent"] == expected_intent, message


def test_unsupported_question_uses_fallback():
    train_intent_model()
    message = "How do I reverse time on Mars?"
    prediction = predict_intent(message)
    response = generate_response(message, prediction["intent"], prediction["confidence"])
    assert response["intent"] in {"unknown", "clarification"}
    assert "rephrase" in response["text"].lower() or "not certain" in response["text"].lower()


def test_saved_model_is_rejected_when_dataset_has_changed(tmp_path, monkeypatch):
    train_intent_model()
    changed_dataset = tmp_path / "intents.json"
    changed_dataset.write_text('{"intents": []}', encoding="utf-8")
    monkeypatch.setattr(nlp_engine, "INTENTS_PATH", changed_dataset)

    with pytest.raises(ValueError, match="different intent dataset"):
        nlp_engine.load_saved_model()
