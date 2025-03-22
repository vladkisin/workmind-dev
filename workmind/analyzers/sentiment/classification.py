import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
from workmind.analyzers.sentiment.base import SentimentAnalyzerBase
from workmind.analyzers.constants import BaseSentiment

MAX_SEQUENCE_LENGTH: int = 512


class ClassificationSentimentAnalyzer(SentimentAnalyzerBase):
    """
    A sentiment analyzer that uses a classification model for sentiment analysis.
    Example: "textattack/bert-base-uncased-SST-2".
    """

    def __init__(
        self,
        model_name: str,
        class_labels: Optional[List[str]] = None,
        batch_size: int = 16,
        hypothesis_template: Optional[str] = None,
    ) -> None:
        """
        Initialize the classification sentiment analyzer.

        Parameters:
            model_name (str): Model name or path.
            class_labels (Optional[List[str]]): List of sentiment labels.
            batch_size (int): Batch size for inference.
            hypothesis_template (Optional[str]): Hypothesis template if applicable.
        """
        super().__init__(model_name, class_labels, batch_size, hypothesis_template)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        if self.class_labels is None:
            self.class_labels = [BaseSentiment.NEGATIVE, BaseSentiment.POSITIVE]

    def infer_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """
        Perform inference on a batch of texts using the classification model.

        Parameters:
            batch (List[str]): List of input texts.

        Returns:
            List[Dict[str, Any]]: List of predictions containing text, predicted sentiment, and probabilities.
        """
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        predictions: List[Dict[str, Any]] = []
        for text, class_idx, prob in zip(
            batch, predicted_classes.tolist(), probabilities
        ):
            sentiment = self.class_labels[class_idx]
            predictions.append(
                {
                    "text": text,
                    "predicted_sentiment": sentiment,
                    "probabilities": prob.tolist(),
                }
            )
        return predictions
