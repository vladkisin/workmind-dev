import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from workmind.analyzers.constants import BaseSentiment


class SentimentAnalyzerBase(ABC):
    """
    Abstract base class for sentiment analyzers.

    This class defines the common interface (batchify, predict, infer_batch) for all sentiment analyzers.
    """

    def __init__(
        self,
        model_name: str,
        class_labels: Optional[List[str]] = None,
        batch_size: int = 16,
        hypothesis_template: Optional[str] = None,
    ) -> None:
        """
        Initialize the sentiment analyzer base.

        Parameters:
            model_name (str): Model name or path.
            class_labels (Optional[List[str]]): List of sentiment labels.
            batch_size (int): Batch size for inference.
            hypothesis_template (Optional[str]): Template for generating hypotheses.
        """
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.class_labels = class_labels
        self.hypothesis_template: str = hypothesis_template or "This text is {}."

    def batchify(self, data: List[Any]) -> List[List[Any]]:
        """
        Split data into batches of size `batch_size`.

        Parameters:
            data (List[Any]): List of items.

        Returns:
            List[List[Any]]: List of batches.
        """
        return [
            data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)
        ]

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiments for a list of texts.

        Parameters:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, Any]]: List of prediction dictionaries.
        """
        all_predictions: List[Dict[str, Any]] = []
        num_batches: int = (len(texts) + self.batch_size - 1) // self.batch_size
        logging_step_den: int = max(num_batches // 10, 1)
        for batch_num, batch in enumerate(self.batchify(texts), start=1):
            if batch_num % logging_step_den == 0:
                print(f"Processing batch {batch_num}/{num_batches}...")
            batch_predictions = self.infer_batch(batch)
            all_predictions.extend(batch_predictions)
        return all_predictions

    def is_negative(self, text: str) -> bool:
        """
        Check if a given text is classified as negative.

        Parameters:
            text (str): Input text.

        Returns:
            bool: True if negative, False otherwise.
        """
        pred = self.predict([text])[0]
        return pred["predicted_sentiment"] == BaseSentiment.NEGATIVE

    @abstractmethod
    def infer_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """
        Abstract method to perform inference on a batch of texts.

        Parameters:
            batch (List[str]): List of input texts.

        Returns:
            List[Dict[str, Any]]: List of predictions.
        """
        pass
