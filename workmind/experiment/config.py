from dataclasses import dataclass
from typing import Optional

from workmind.analyzers.constants import BaseSentiment

DEFAULT_BATCH_SIZE: int = 32
SMALL_BATCH_SIZE: int = 16


@dataclass(frozen=True)
class SentimentInferenceType:
    """
    Dataclass for sentiment inference types.
    """

    NLI: str = "nli"
    CLASSIFICATION: str = "classification"
    LLM: str = "llm"


@dataclass(frozen=True)
class ExperimentNames:
    """
    Dataclass containing experiment name constants.
    """

    BERT_CLS_PRETRAINED: str = "BERT CLS Pretrained"
    XLNET_CLS_PRETRAINED: str = "XLNet CLS Pretrained"
    ROBERTA_LARGE_CLS_PRETRAINED: str = "Roberta Large CLS Pretrained"
    BERT_SENTIMENT_PRETRAINED_5_CLASSES: str = "BERT Sentiment Pretrained 5 classes"
    ROBERTA_LARGE_NLI: str = "Roberta Large NLI"
    DEBERTA_V3_LARGE_NLI: str = "Deberta-v3 Large NLI"


@dataclass(frozen=True)
class HypothesisTemplates:
    """
    Dataclass for hypothesis templates.
    """

    DEFAULT_NLI: str = "This text conveys a {} sentiment."
    NONE: Optional[str] = None


@dataclass(frozen=True)
class ConfigKeys:
    """
    Dataclass for configuration keys.
    """

    INFERENCE_TYPE: str = "inference_type"
    CLASS_LABELS: str = "class_labels"
    BATCH_SIZE: str = "batch_size"
    HYPOTHESIS_TEMPLATE: str = "hypothesis_template"
    EXPERIMENT_NAME: str = "experiment_name"


MODELS_CONFIG: dict = {
    "textattack/bert-base-uncased-SST-2": {
        ConfigKeys.INFERENCE_TYPE: SentimentInferenceType.CLASSIFICATION,
        ConfigKeys.CLASS_LABELS: [BaseSentiment.NEGATIVE, BaseSentiment.POSITIVE],
        ConfigKeys.BATCH_SIZE: DEFAULT_BATCH_SIZE,
        ConfigKeys.HYPOTHESIS_TEMPLATE: HypothesisTemplates.NONE,
        ConfigKeys.EXPERIMENT_NAME: ExperimentNames.BERT_CLS_PRETRAINED,
    },
    "dipawidia/xlnet-base-cased-product-review-sentiment-analysis": {
        ConfigKeys.INFERENCE_TYPE: SentimentInferenceType.CLASSIFICATION,
        ConfigKeys.CLASS_LABELS: [BaseSentiment.NEGATIVE, BaseSentiment.POSITIVE],
        ConfigKeys.BATCH_SIZE: DEFAULT_BATCH_SIZE,
        ConfigKeys.HYPOTHESIS_TEMPLATE: HypothesisTemplates.NONE,
        ConfigKeys.EXPERIMENT_NAME: ExperimentNames.XLNET_CLS_PRETRAINED,
    },
    "siebert/sentiment-roberta-large-english": {
        ConfigKeys.INFERENCE_TYPE: SentimentInferenceType.CLASSIFICATION,
        ConfigKeys.CLASS_LABELS: [BaseSentiment.NEGATIVE, BaseSentiment.POSITIVE],
        ConfigKeys.BATCH_SIZE: DEFAULT_BATCH_SIZE,
        ConfigKeys.HYPOTHESIS_TEMPLATE: HypothesisTemplates.NONE,
        ConfigKeys.EXPERIMENT_NAME: ExperimentNames.ROBERTA_LARGE_CLS_PRETRAINED,
    },
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        ConfigKeys.INFERENCE_TYPE: SentimentInferenceType.CLASSIFICATION,
        ConfigKeys.CLASS_LABELS: [
            BaseSentiment.NEGATIVE,
            BaseSentiment.NEGATIVE,
            BaseSentiment.NEUTRAL,
            BaseSentiment.POSITIVE,
            BaseSentiment.POSITIVE,
        ],
        ConfigKeys.BATCH_SIZE: DEFAULT_BATCH_SIZE,
        ConfigKeys.HYPOTHESIS_TEMPLATE: HypothesisTemplates.NONE,
        ConfigKeys.EXPERIMENT_NAME: ExperimentNames.BERT_SENTIMENT_PRETRAINED_5_CLASSES,
    },
    "roberta-large-mnli": {
        ConfigKeys.INFERENCE_TYPE: SentimentInferenceType.NLI,
        ConfigKeys.CLASS_LABELS: [
            BaseSentiment.NEGATIVE,
            BaseSentiment.NEUTRAL,
            BaseSentiment.POSITIVE,
        ],
        ConfigKeys.BATCH_SIZE: SMALL_BATCH_SIZE,
        ConfigKeys.HYPOTHESIS_TEMPLATE: HypothesisTemplates.DEFAULT_NLI,
        ConfigKeys.EXPERIMENT_NAME: ExperimentNames.ROBERTA_LARGE_NLI,
    },
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": {
        ConfigKeys.INFERENCE_TYPE: SentimentInferenceType.NLI,
        ConfigKeys.CLASS_LABELS: [
            BaseSentiment.NEGATIVE,
            BaseSentiment.NEUTRAL,
            BaseSentiment.POSITIVE,
        ],
        ConfigKeys.BATCH_SIZE: SMALL_BATCH_SIZE,
        ConfigKeys.HYPOTHESIS_TEMPLATE: HypothesisTemplates.DEFAULT_NLI,
        ConfigKeys.EXPERIMENT_NAME: ExperimentNames.DEBERTA_V3_LARGE_NLI,
    },
}
