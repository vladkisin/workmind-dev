from typing import List

from workmind.analyzers.sentiment.classification import ClassificationSentimentAnalyzer
from workmind.analyzers.sentiment.nli import NLISentimentAnalyzer
from workmind.analyzers.sentiment.llm import LLMSentimentAnalyzer
from workmind.analyzers.constants import SentimentInferenceType


def get_analyzer(
    inference_type: str,
    model_name: str,
    class_labels: List[str] = None,
    batch_size: int = 16,
    hypothesis_template: str = None,
):
    """
    Factory method to create a sentiment analyzer based on the inference_type.

    :param inference_type: One of ["classification", "nli", "llm"].
    :param model_name: Model name or path from Hugging Face.
    :param class_labels: Custom list of labels if desired.
    :param batch_size: Inference batch size.
    :param hypothesis_template: For NLI or LLM, a prompt/template string.
    :return: An instance of ClassificationSentimentAnalyzer, NLISentimentAnalyzer, or LLMSentimentAnalyzer.
    """
    inference_type = inference_type.lower()

    if inference_type == SentimentInferenceType.CLASSIFICATION:
        return ClassificationSentimentAnalyzer(
            model_name=model_name,
            class_labels=class_labels,
            batch_size=batch_size,
            hypothesis_template=hypothesis_template,
        )
    elif inference_type == SentimentInferenceType.NLI:
        return NLISentimentAnalyzer(
            model_name=model_name,
            class_labels=class_labels,
            batch_size=batch_size,
            hypothesis_template=hypothesis_template,
        )
    elif inference_type == SentimentInferenceType.LLM:
        return LLMSentimentAnalyzer(
            model_name=model_name,
            class_labels=class_labels,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"""Unknown inference type: {inference_type}. 
        Choose from {SentimentInferenceType.CLASSIFICATION}, {SentimentInferenceType.LLM},{SentimentInferenceType.NLI}"""
        )
