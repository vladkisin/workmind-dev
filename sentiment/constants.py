from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentInferenceType:
    NLI = "nli"
    CLASSIFICATION = "classification"
    LLM = "llm"
