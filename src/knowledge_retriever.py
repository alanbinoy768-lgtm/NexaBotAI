import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import KNOWLEDGE_PATH
from src.preprocessing import preprocess_text


def load_knowledge_base(path: str | Path = KNOWLEDGE_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        return data.get("entries", [])
    return data


def retrieve_answer(question: str, threshold: float = 0.18, knowledge_path: str | Path = KNOWLEDGE_PATH) -> Optional[Dict[str, Any]]:
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")
    entries = load_knowledge_base(knowledge_path)
    if not entries:
        return None

    documents = []
    for entry in entries:
        query_text = f"{entry.get('question', '')} {entry.get('answer', '')}"
        documents.append(preprocess_text(query_text))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([preprocess_text(question), *documents])
    similarities = cosine_similarity(matrix[0], matrix[1:])[0]
    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index])

    if best_score < threshold:
        return None

    best_entry = entries[best_index]
    return {
        "question": best_entry.get("question"),
        "answer": best_entry.get("answer"),
        "score": round(best_score, 4),
        "category": best_entry.get("category", "general"),
    }
