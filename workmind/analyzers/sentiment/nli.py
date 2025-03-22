import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from workmind.analyzers.sentiment.base import SentimentAnalyzerBase
from workmind.analyzers.constants import BaseSentiment


class NLISentimentAnalyzer(SentimentAnalyzerBase):
    """
    Uses an NLI-based model for zero-shot sentiment analysis.
    E.g., "roberta-large-mnli", "microsoft/deberta-v3-base-mnli", etc.
    """

    def __init__(
        self, model_name, class_labels=None, batch_size=16, hypothesis_template=None
    ):
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
        self.entailment_index = self._get_entailment_index()

    def _get_entailment_index(self):
        """
        Determine the entailment index based on the model config if possible.
        Fallback to heuristics: DeBERTa=0, others=2.
        """
        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            for idx, label_name in self.model.config.id2label.items():
                if "entail" in label_name.lower():
                    return idx
        if "deberta" in self.model_name.lower():
            return 0
        return 2

    def infer_batch(self, batch):
        """
        For each label in self.class_labels, build a hypothesis using self.hypothesis_template,
        compute the 'entailment' probability, and pick the highest one.
        """
        predictions = []
        for text in batch:
            hypotheses = [
                self.hypothesis_template.format(label) for label in self.class_labels
            ]

            inputs = self.tokenizer(
                [text] * len(hypotheses),
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            entailment_scores = probabilities[:, self.entailment_index]

            sentiment_scores = dict(zip(self.class_labels, entailment_scores.tolist()))
            predicted_label = max(sentiment_scores, key=sentiment_scores.get)

            predictions.append(
                {
                    "text": text,
                    "predicted_sentiment": predicted_label,
                    "sentiment_scores": sentiment_scores,
                }
            )
        return predictions
