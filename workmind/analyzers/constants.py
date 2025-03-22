from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentInferenceType:
    NLI: str = "nli"
    CLASSIFICATION: str = "classification"
    LLM: str = "llm"


@dataclass(frozen=True)
class BaseSentiment:
    NEGATIVE: str = "negative"
    NEUTRAL: str = "neutral"
    POSITIVE: str = "positive"
