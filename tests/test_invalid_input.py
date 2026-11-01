import pytest

from src.nlp_engine import predict_intent


def test_empty_input_raises():
    with pytest.raises(ValueError):
        predict_intent("   ")
