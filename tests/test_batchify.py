import pytest
from workmind.analyzers.sentiment.base import SentimentAnalyzerBase


class DummySentimentAnalyzer(SentimentAnalyzerBase):
    def infer_batch(self, batch):
        return [
            {
                "text": text,
                "predicted_sentiment": "neutral",
                "probabilities": [0.33, 0.33, 0.34],
            }
            for text in batch
        ]


def test_batchify():
    dummy = DummySentimentAnalyzer("dummy-model", batch_size=2)
    data = ["text1", "text2", "text3", "text4", "text5"]
    batches = list(dummy.batchify(data))
    assert len(batches) == 3
    assert batches[0] == ["text1", "text2"]
    assert batches[1] == ["text3", "text4"]
    assert batches[2] == ["text5"]


def test_predict():
    dummy = DummySentimentAnalyzer("dummy-model", batch_size=2)
    texts = ["Hello world.", "Testing batchify.", "Another sentence."]
    predictions = dummy.predict(texts)
    assert len(predictions) == len(texts)
    for pred in predictions:
        assert "text" in pred
        assert "predicted_sentiment" in pred
        assert "probabilities" in pred
