import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
from workmind.analyzers.sentiment.base import SentimentAnalyzerBase
from workmind.analyzers.constants import BaseSentiment

MAX_SEQUENCE_LENGTH_NLI: int = 512


class NLISentimentAnalyzer(SentimentAnalyzerBase):
    """
    An NLI-based sentiment analyzer that uses a natural language inference model to predict sentiment.
    """

    def __init__(
        self,
        model_name: str,
        class_labels: Optional[List[str]] = None,
        batch_size: int = 16,
        hypothesis_template: Optional[str] = None,
    ) -> None:
        """
        Initialize the NLI sentiment analyzer.

        Parameters:
            model_name (str): Model name or path.
            class_labels (Optional[List[str]]): List of sentiment labels.
            batch_size (int): Batch size for inference.
            hypothesis_template (Optional[str]): Template for generating NLI hypotheses.
        """
        super().__init__(model_name, class_labels, batch_size, hypothesis_template)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        if self.class_labels is None:
            self.class_labels = [
                BaseSentiment.NEGATIVE,
                BaseSentiment.NEUTRAL,
                BaseSentiment.POSITIVE,
            ]
        self.entailment_index: int = self._get_entailment_index()

    def _get_entailment_index(self) -> int:
        """
        Determine the entailment index based on the model configuration.

        Returns:
            int: The entailment index.
        """
        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            for idx, label_name in self.model.config.id2label.items():
                if "entail" in label_name.lower():
                    return idx
        if "deberta" in self.model_name.lower():
            return 0
        return 2

    def infer_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """
        Perform zero-shot sentiment inference using NLI for a batch of texts.

        Parameters:
            batch (List[str]): List of input texts.

        Returns:
            List[Dict[str, Any]]: List of predictions with keys 'text', 'predicted_sentiment', and 'sentiment_scores'.
        """
        predictions: List[Dict[str, Any]] = []
        for text in batch:
            hypotheses: List[str] = [
                self.hypothesis_template.format(label) for label in self.class_labels
            ]
            inputs = self.tokenizer(
                [text] * len(hypotheses),
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH_NLI,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            entailment_scores = probabilities[:, self.entailment_index]
            sentiment_scores: Dict[str, float] = dict(
                zip(self.class_labels, entailment_scores.tolist())
            )
            predicted_label: str = max(sentiment_scores, key=sentiment_scores.get)
            predictions.append(
                {
                    "text": text,
                    "predicted_sentiment": predicted_label,
                    "sentiment_scores": sentiment_scores,
                }
            )
        return predictions
