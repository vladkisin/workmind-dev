import pytest
from workmind.analyzers import get_analyzer
from workmind.analyzers.sentiment.classification import ClassificationSentimentAnalyzer
from workmind.analyzers.sentiment.nli import NLISentimentAnalyzer
from workmind.analyzers.sentiment.llm import LLMSentimentAnalyzer


def test_get_analyzer_classification():
    analyzer = get_analyzer(
        inference_type="classification",
        model_name="dummy-model",
        class_labels=["negative", "positive"],
        batch_size=8,
    )
    assert isinstance(analyzer, ClassificationSentimentAnalyzer)


def test_get_analyzer_nli():
    analyzer = get_analyzer(
        inference_type="nli",
        model_name="dummy-model",
        class_labels=["negative", "neutral", "positive"],
        batch_size=8,
        hypothesis_template="This text conveys a {} sentiment.",
    )
    assert isinstance(analyzer, NLISentimentAnalyzer)


def test_get_analyzer_llm():
    analyzer = get_analyzer(
        inference_type="llm",
        model_name="dummy-model",
        class_labels=["negative", "neutral", "positive"],
        batch_size=8,
        hypothesis_template="This text conveys a {} sentiment.",
    )
    assert isinstance(analyzer, LLMSentimentAnalyzer)


def test_get_analyzer_invalid_type():
    with pytest.raises(ValueError):
        get_analyzer(
            inference_type="invalid",
            model_name="dummy-model",
        )
