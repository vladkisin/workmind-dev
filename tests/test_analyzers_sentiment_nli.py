import pytest
from workmind.analyzers.sentiment.nli import NLISentimentAnalyzer
from workmind.analyzers.constants import BaseSentiment


def test_nli_infer_batch():
    analyzer = NLISentimentAnalyzer(
        model_name="dummy-model",
        class_labels=[
            BaseSentiment.NEGATIVE,
            BaseSentiment.NEUTRAL,
            BaseSentiment.POSITIVE,
        ],
        hypothesis_template="This text conveys a {} sentiment.",
    )
    batch = ["This is a test.", "Another test sentence."]
    predictions = analyzer.infer_batch(batch)
    assert len(predictions) == len(batch)
    for pred in predictions:
        assert "text" in pred
        assert "predicted_sentiment" in pred
        assert "sentiment_scores" in pred
