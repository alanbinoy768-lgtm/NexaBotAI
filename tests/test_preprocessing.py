from src.preprocessing import clean_text, preprocess_text, tokenize


def test_clean_text_removes_symbols_and_lowercases():
    text = "Hello, WORLD!!!"
    cleaned = clean_text(text)
    assert cleaned == "hello world"


def test_preprocess_text_preserves_meaningful_short_words():
    text = "How are you?"
    prepared = preprocess_text(text)
    assert prepared == "how are you"


def test_preprocess_text_expands_contractions():
    assert preprocess_text("What's up?") == "what is up"


def test_tokenize_returns_words():
    tokens = tokenize("How are you doing today?")
    assert isinstance(tokens, list)
    assert "how" in tokens
