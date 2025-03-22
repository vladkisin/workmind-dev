from typing import Dict
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from adapters import AdapterTrainer


class WeightedLossTrainer(AdapterTrainer):
    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        class_weights = torch.tensor([0.5, 0.3, 0.2]).to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def default_compute_metrics(eval_prediction) -> Dict[str, float]:
    """
    Default compute metrics function for classification heads.
    """
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    accuracy = (preds == labels).mean()

    negative_indices = labels == 0
    acc_neg = (
        (preds[negative_indices] == labels[negative_indices]).mean()
        if negative_indices.sum() > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "acc_neg": acc_neg,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
