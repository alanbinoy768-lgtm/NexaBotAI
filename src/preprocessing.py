import re
import unicodedata
from typing import List

from nltk.tokenize import word_tokenize

from src.config import ensure_nltk_data

ensure_nltk_data()
CONTRACTIONS = {
    "what's": "what is",
    "who's": "who is",
    "how's": "how is",
    "where's": "where is",
    "when's": "when is",
    "i'm": "i am",
    "you're": "you are",
    "it's": "it is",
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
}


def normalize_whitespace(text: str) -> str:
    if text is None:
        raise TypeError("Text input cannot be None")
    return " ".join(str(text).split())


def clean_text(text: str) -> str:
    if text is None:
        raise TypeError("Text input cannot be None")
    text = unicodedata.normalize("NFKC", str(text)).lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)


def tokenize(text: str) -> List[str]:
    cleaned = clean_text(text)
    try:
        return word_tokenize(cleaned)
    except LookupError:
        return cleaned.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    from nltk.corpus import stopwords

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()
    return [token for token in tokens if token not in stop_words and len(token) > 1]


def preprocess_text(text: str, remove_stopwords_flag: bool = False) -> str:
    tokens = tokenize(text)
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)
