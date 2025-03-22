import pytest
from workmind.experiment.utils import (
    clean_text,
    calculate_user_level_metrics,
    compute_bleu,
)


def test_clean_text():
    text = "   This   is    a  test.  "
    cleaned = clean_text(text)
    assert cleaned == "This is a test."


def test_calculate_user_level_metrics():
    user_ids = ["u1", "u1", "u2", "u3", "u3"]
    predicted = ["negative", "positive", "positive", "negative", "negative"]
    real = ["negative", "positive", "positive", "positive", "negative"]
    metrics = calculate_user_level_metrics(user_ids, predicted, real)
    assert "macro avg" in metrics
    assert 0 <= metrics.get("accuracy", 0) <= 1


def test_compute_bleu():
    candidate = ["This is a test sentence."]
    reference = ["This is a test sentence."]
    bleu_scores = compute_bleu(candidate, reference)
    assert bleu_scores[0] > 0.9
