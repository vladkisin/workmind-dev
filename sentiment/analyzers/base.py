import torch
from abc import ABC, abstractmethod

from sentiment.constants import BaseSentiment


class SentimentAnalyzerBase(ABC):
    """
    Abstract Base Class for sentiment analysis.
    It defines the common interface: batchify, predict, infer_batch.
    """

    def __init__(self,
                 model_name: str,
                 class_labels=None,
                 batch_size=16,
                 hypothesis_template=None):
        """
        :param model_name: Name of the model from Hugging Face.
        :param class_labels: List of class labels (e.g., ["negative", "positive"] or ["neg","neu","pos"]).
        :param batch_size: Batch size for inference.
        :param hypothesis_template: If applicable, template for generating NLI hypotheses.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.class_labels = class_labels
        self.hypothesis_template = hypothesis_template or "This text is {}."

    def batchify(self, data):
        """Split data into chunks of size batch_size."""
        for i in range(0, len(data), self.batch_size):
            yield data[i:i + self.batch_size]

    def predict(self, texts):
        """
        Predict sentiments for a list of texts.
        Returns a list of dictionaries with:
          - "text"
          - "predicted_sentiment"
          - probabilities or sentiment_scores (may vary by subclass)
        """
        all_predictions = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logging_step_den = num_batches // 10
        for batch_num, batch in enumerate(self.batchify(texts), start=1):
            if batch_num % logging_step_den == 0:
                print(f"Processing batch {batch_num}/{num_batches}...")
            batch_predictions = self.infer_batch(batch)
            all_predictions.extend(batch_predictions)

        return all_predictions

    def is_negative(self, text):
        pred = self.predict([text])[0]
        return pred["predicted_sentiment"] == BaseSentiment.NEGATIVE

    @abstractmethod
    def infer_batch(self, batch):
        """Subclasses implement their specific inference logic."""
        pass
