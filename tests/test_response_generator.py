from src.response_generator import generate_response


def test_generate_response_for_greeting():
    result = generate_response("hello", "greeting", 0.78)
    assert "Hello" in result["text"] or "Hi" in result["text"]
    assert result["intent"] == "greeting"


def test_generate_response_for_unknown_question():
    result = generate_response("ask me about a totally weird thing", "unknown", 0.10)
    assert "rephrase" in result["text"].lower() or "clearer" in result["text"].lower()


def test_supported_knowledge_answer_is_not_replaced_by_generic_intent_response():
    result = generate_response(
        "What is machine learning?",
        "technology",
        0.9,
        knowledge_answer="Machine learning lets systems learn patterns from data.",
    )
    assert result["text"] == "Machine learning lets systems learn patterns from data."


def test_how_are_you_receives_a_conversational_greeting():
    result = generate_response("How are you?", "greeting", 0.8)
    assert "thanks for asking" in result["text"].lower()
