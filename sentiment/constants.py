from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentInferenceType:
    NLI = "nli"
    CLASSIFICATION = "classification"
    LLM = "llm"


@dataclass(frozen=True)
class BaseSentiment:
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
