import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment.analyzers.base import SentimentAnalyzerBase


class LLMSentimentAnalyzer(SentimentAnalyzerBase):
    pass
