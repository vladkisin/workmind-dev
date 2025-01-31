from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def default_compute_metrics(eval_prediction) -> Dict[str, float]:
    """
    Default compute metrics function for classification heads.
    """
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    accuracy = (preds == labels).mean()

    negative_indices = (labels == 0)
    acc_neg = (preds[negative_indices] == labels[negative_indices]).mean() if negative_indices.sum() > 0 else 0.0

    return {
        "accuracy": accuracy,
        "acc_neg": acc_neg,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
