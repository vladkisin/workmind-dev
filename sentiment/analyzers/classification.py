import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment.analyzers.base import SentimentAnalyzerBase
from sentiment.constants import BaseSentiment


class ClassificationSentimentAnalyzer(SentimentAnalyzerBase):
    """
    Uses a classification (fine-tuned) model for sentiment.
    E.g., "textattack/bert-base-uncased-SST-2".
    """

    def __init__(self, model_name, class_labels=None, batch_size=16, hypothesis_template=None):
        super().__init__(model_name, class_labels, batch_size, hypothesis_template)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        if self.class_labels is None:
            self.class_labels = [BaseSentiment.NEGATIVE,
                                 BaseSentiment.POSITIVE]

    def infer_batch(self, batch):
        """Perform standard classification inference, returning predicted sentiment + probabilities."""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        predictions = []
        for text, class_idx, prob in zip(batch, predicted_classes.tolist(), probabilities):
            sentiment = self.class_labels[class_idx]
            predictions.append({
                "text": text,
                "predicted_sentiment": sentiment,
                "probabilities": prob.tolist()
            })
        return predictions
