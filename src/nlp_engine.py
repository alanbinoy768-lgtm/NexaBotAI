import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import INTENTS_PATH, METRICS_PATH, MODEL_PATH, VECTORIZER_PATH, ensure_directories
from src.preprocessing import preprocess_text


def load_intent_dataset(path: str | Path = INTENTS_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "intents" not in data:
        raise ValueError("Intent dataset must contain an 'intents' key.")
    return data


def create_training_examples(intent_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for intent in intent_data["intents"]:
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        if tag is None or not patterns:
            continue
        for pattern in patterns:
            texts.append(preprocess_text(str(pattern)))
            labels.append(tag)
    if not texts:
        raise ValueError("No training examples found in the intent dataset.")
    return texts, labels


def build_model() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                strip_accents="unicode",
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1500, random_state=42, class_weight="balanced", C=20.0)),
        ]
    )
    return pipeline


def train_intent_model(dataset_path: str | Path = INTENTS_PATH) -> Dict[str, Any]:
    ensure_directories()
    dataset = load_intent_dataset(dataset_path)
    X, y = create_training_examples(dataset)
    if len(set(y)) < 2:
        raise ValueError("At least two classes are required for supervised training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(set(y))).tolist(),
        "classes": sorted(set(y)),
        "training_examples": len(X),
        "train_examples": len(X_train),
        "test_examples": len(X_test),
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model.named_steps["tfidf"], VECTORIZER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def load_saved_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not METRICS_PATH.exists():
        raise ValueError("Model metadata is missing; retrain the intent model.")
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    current_hash = hashlib.sha256(INTENTS_PATH.read_bytes()).hexdigest()
    if metadata.get("dataset_sha256") != current_hash:
        raise ValueError("The saved model was trained on a different intent dataset; retrain it.")
    return joblib.load(MODEL_PATH)


def predict_intent(text: str, model: Pipeline | None = None) -> Dict[str, Any]:
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")
    if model is None:
        model = load_saved_model()
    cleaned = preprocess_text(text)
    transformed = model.named_steps["tfidf"].transform([cleaned])
    if transformed.nnz == 0:
        return {"intent": "unknown", "confidence": 0.0, "text": cleaned}
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = float(max(probabilities))
    return {"intent": prediction, "confidence": round(confidence, 4), "text": cleaned}


def get_model_metadata() -> Dict[str, Any]:
    metadata = {"model_path": str(MODEL_PATH), "vectorizer_path": str(VECTORIZER_PATH), "supports_intents": []}
    if INTENTS_PATH.exists():
        dataset = load_intent_dataset(INTENTS_PATH)
        metadata["supports_intents"] = [intent["tag"] for intent in dataset["intents"]]
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metadata["metrics"] = json.load(file)
    else:
        metadata["metrics"] = {"status": "not_trained_yet"}
    return metadata
