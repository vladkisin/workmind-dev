from typing import Dict
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from adapters import AdapterTrainer


class WeightedLossTrainer(AdapterTrainer):
    """
    A trainer that uses a weighted loss for imbalanced classification.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: int = None,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """
        Compute loss with weighted cross-entropy.

        Parameters:
            model (torch.nn.Module): The model.
            inputs (Dict[str, torch.Tensor]): Input tensors.
            num_items_in_batch (int): Number of items in batch.
            return_outputs (bool): Whether to return outputs.

        Returns:
            torch.Tensor: Computed loss.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        class_weights = torch.tensor([0.5, 0.3, 0.2]).to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def default_compute_metrics(eval_prediction: pd.DataFrame) -> Dict[str, float]:
    """
    Compute default metrics (accuracy, f1, precision, recall) for classification.

    Parameters:
        eval_prediction (Any): Object with predictions and label_ids.

    Returns:
        Dict[str, float]: Metrics dictionary.
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
