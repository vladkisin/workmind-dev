import pytest
from workmind.analyzers.sentiment.llm import LLMSentimentAnalyzer
from workmind.analyzers.constants import BaseSentiment


def test_llm_infer_batch():
    analyzer = LLMSentimentAnalyzer(
        model_name="dummy-model",
        class_labels=[
            BaseSentiment.NEGATIVE,
            BaseSentiment.NEUTRAL,
            BaseSentiment.POSITIVE,
        ],
        batch_size=2,
    )
    batch = ["Test input one.", "Test input two."]
    predictions = analyzer.infer_batch(batch)
    assert len(predictions) == len(batch)
    for pred in predictions:
        assert "text" in pred
        assert "predicted_sentiment" in pred
