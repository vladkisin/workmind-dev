import time
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix
from workmind.experiment.utils import calculate_user_level_metrics


class SentimentExperiment:
    """
    W&B-based experiment wrapper for logging and evaluating sentiment analysis experiments.
    """

    def __init__(self, analyzer: Any, experiment_name: str, true_labels: Optional[List[str]] = None, log_predictions: bool = False, project_name: str = "workmind") -> None:
        """
        Initialize the sentiment experiment.

        Parameters:
            analyzer (Any): A sentiment analyzer with a predict method.
            experiment_name (str): Name of the experiment.
            true_labels (Optional[List[str]]): Ground-truth labels.
            log_predictions (bool): Whether to log predictions.
            project_name (str): W&B project name.
        """
        self.analyzer: Any = analyzer
        self.project_name: str = project_name
        self.experiment_name: str = experiment_name
        self.true_labels: Optional[List[str]] = true_labels
        self.log_predictions: bool = log_predictions
        self.run: Optional[Any] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "SentimentExperiment":
        self.start_time = time.time()
        self.run = wandb.init(project=self.project_name, name=self.experiment_name, reinit=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        wandb.log({"total_time_seconds": elapsed})
        self.run.finish()

    def evaluate(self, texts: List[str], user_ids: Optional[List[str]] = None) -> None:
        """
        Run sentiment predictions and log results to W&B.

        Parameters:
            texts (List[str]): List of input texts.
            user_ids (Optional[List[str]]): List of user IDs for aggregation.
        """
        predictions = self.analyzer.predict(texts)
        wandb.config.update({"model_name": self.analyzer.model_name}, allow_val_change=True)
        if hasattr(self.analyzer, "mode"):
            wandb.config.update({"mode": self.analyzer.mode}, allow_val_change=True)
        if hasattr(self.analyzer, "class_labels"):
            wandb.config.update({"class_labels": self.analyzer.class_labels}, allow_val_change=True)
        if hasattr(self.analyzer, "hypothesis_template"):
            wandb.config.update({"hypothesis_template": self.analyzer.hypothesis_template}, allow_val_change=True)
        if self.log_predictions:
            preds_file = "predictions.txt"
            with open(preds_file, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(f"Text: {p['text']}\nPredicted: {p['predicted_sentiment']}\n----\n")
            wandb.save(preds_file)
        if self.true_labels:
            predicted_labels = [p["predicted_sentiment"] for p in predictions]
            self.log_metrics(self.true_labels, predicted_labels, user_ids)
    @staticmethod
    def log_metrics(true_labels: List[str], predicted_labels: List[str], user_ids: Optional[List[str]] = None) -> None:
        """
        Compute and log sentiment metrics and confusion matrix to W&B.

        Parameters:
            true_labels (List[str]): True sentiment labels.
            predicted_labels (List[str]): Predicted sentiment labels.
            user_ids (Optional[List[str]]): User IDs for aggregation.
        """
        report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
        wandb.log({"classification_report": report_dict})
        wandb.log({
            "precision_macro": report_dict["macro avg"]["precision"],
            "recall_macro": report_dict["macro avg"]["recall"],
            "f1_macro": report_dict["macro avg"]["f1-score"],
        })
        if "accuracy" in report_dict:
            wandb.log({"accuracy": report_dict["accuracy"]})
        neg_scores = report_dict.get("negative")
        if neg_scores:
            wandb.log({
                "precision_negative": neg_scores["precision"],
                "recall_negative": neg_scores["recall"],
                "f1_negative": neg_scores["f1-score"],
            })
        if user_ids:
            user_level_metrics = calculate_user_level_metrics(user_ids, predicted_labels, true_labels)
            wandb.log({
                "precision_user_macro": user_level_metrics["macro avg"]["precision"],
                "recall_user_macro": user_level_metrics["macro avg"]["recall"],
                "f1_user_macro": user_level_metrics["macro avg"]["f1-score"],
            })
            neg_user_scores = user_level_metrics.get("negative")
            wandb.log({
                "precision_user_negative": neg_user_scores["precision"],
                "recall_user_negative": neg_user_scores["recall"],
                "f1_user_negative": neg_user_scores["f1-score"],
            })
        exclude_keys = {"accuracy", "macro avg", "weighted avg"}
        labels = [k for k in report_dict.keys() if k not in exclude_keys]
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        cm_png = "confusion_matrix.png"
        fig.savefig(cm_png, bbox_inches="tight")
        wandb.log({"confusion_matrix": wandb.Image(cm_png)})
        plt.close(fig)
