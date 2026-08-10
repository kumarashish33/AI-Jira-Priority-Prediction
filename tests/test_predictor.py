import pytest

from src.predictor import predict_priority


def test_prediction_returns_valid_priority():
    priority, confidence, probabilities = predict_priority(
        "Application crashes after login"
    )

    assert priority in ["Highest", "High", "Medium", "Low"]


def test_confidence_range():
    _, confidence, _ = predict_priority("Application crashes after login")

    assert 0.0 <= confidence <= 1.0


def test_invalid_input_type():
    with pytest.raises(TypeError):
        predict_priority(123)


def test_empty_input():
    with pytest.raises(ValueError):
        predict_priority("")
