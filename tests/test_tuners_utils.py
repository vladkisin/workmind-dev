import pytest
import numpy as np
from workmind.tuners.utils import default_compute_metrics


class DummyEvalPrediction:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


def test_default_compute_metrics():
    preds = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.2, 0.6]])
    labels = np.array([1, 0, 2])
    eval_pred = DummyEvalPrediction(preds, labels)
    metrics = default_compute_metrics(eval_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
