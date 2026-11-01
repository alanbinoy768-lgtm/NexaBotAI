from src.config import ensure_directories
from src.nlp_engine import get_model_metadata, train_intent_model


def main():
    ensure_directories()
    metrics = train_intent_model()
    print("Model training completed.")
    print(metrics)
    print(get_model_metadata())


if __name__ == "__main__":
    main()
