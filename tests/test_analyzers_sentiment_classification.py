import pytest
from workmind.analyzers.sentiment.classification import ClassificationSentimentAnalyzer
from workmind.analyzers.constants import BaseSentiment


def test_classification_infer_batch():
    analyzer = ClassificationSentimentAnalyzer(
        model_name="dummy-model",
        class_labels=[BaseSentiment.NEGATIVE, BaseSentiment.POSITIVE],
    )
    batch = ["This is a test.", "Another test."]
    predictions = analyzer.infer_batch(batch)
    assert len(predictions) == len(batch)
    for pred in predictions:
        assert "text" in pred
        assert "predicted_sentiment" in pred
        assert "probabilities" in pred
