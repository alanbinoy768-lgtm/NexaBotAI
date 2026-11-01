from pathlib import Path
import os
import nltk

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
INTENTS_PATH = DATA_DIR / "intents.json"
KNOWLEDGE_PATH = DATA_DIR / "knowledge_base.json"
DB_PATH = DATA_DIR / "chat_history.db"
MODEL_PATH = MODELS_DIR / "intent_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.08
OLLAMA_BASE_URL = os.environ.get("NEXABOT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.environ.get("NEXABOT_OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_CONNECT_TIMEOUT = 5
OLLAMA_READ_TIMEOUT = 600
MAX_USER_MESSAGE_CHARS = 16000
MAX_CONTEXT_CHARS = 24000
MAX_CONTEXT_MESSAGES = 20


def ensure_nltk_data():
    resources = ["punkt", "punkt_tab", "stopwords"]
    for resource in resources:
        try:
            if resource.startswith("punkt"):
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def ensure_directories():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    (ROOT_DIR / "assets").mkdir(exist_ok=True)
