from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from workmind.analyzers.sentiment.base import SentimentAnalyzerBase
from workmind.analyzers.constants import BaseSentiment

MAX_SEQUENCE_LENGTH: int = 512


class AdapterClassificationSentimentAnalyzer(SentimentAnalyzerBase):
    """
    Adapter-based sentiment analyzer using a fine-tuned model with an adapter.
    Example: "textattack/bert-base-uncased-SST-2".
    """

    def __init__(
        self,
        model_name: str,
        adapter_name: str,
        class_labels: Optional[List[str]] = None,
        batch_size: int = 16,
        hypothesis_template: Optional[str] = None,
    ) -> None:
        """
        Initialize the adapter-based sentiment analyzer.

        Parameters:
            model_name (str): Model name or path.
            adapter_name (str): Name of the adapter to load.
            class_labels (Optional[List[str]]): List of sentiment labels.
            batch_size (int): Batch size for inference.
            hypothesis_template (Optional[str]): Hypothesis template (if applicable).
        """
        super().__init__(model_name, class_labels, batch_size, hypothesis_template)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        adapter = self.model.load_adapter(adapter_name)
        self.model.set_active_adapters(adapter)
        self.model.to(self.device)
        if self.class_labels is None:
            self.class_labels = [
                BaseSentiment.NEGATIVE,
                BaseSentiment.NEUTRAL,
                BaseSentiment.POSITIVE,
            ]

    def infer_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """
        Perform classification inference on a batch of texts.

        Parameters:
            batch (List[str]): List of input texts.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with keys 'text', 'predicted_sentiment', and 'probabilities'.
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
