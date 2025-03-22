import time
import json
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix
from workmind.experiment.utils import calculate_user_level_metrics


class SentimentExperiment:
    """
    MLflow-based experiment wrapper for logging and evaluating sentiment analysis experiments.
    """

    def __init__(
        self,
        analyzer: Any,
        experiment_name: str,
        true_labels: Optional[List[str]] = None,
        log_predictions: bool = False,
    ) -> None:
        """
        Initialize the sentiment experiment.

        Parameters:
            analyzer (Any): A sentiment analyzer with a predict(texts) method.
            experiment_name (str): Name of the MLflow experiment.
            true_labels (Optional[List[str]]): Ground-truth sentiment labels.
            log_predictions (bool): Whether to log predictions as an artifact.
        """
        self.analyzer: Any = analyzer
        self.experiment_name: str = experiment_name
        self.true_labels: Optional[List[str]] = true_labels
        self.log_predictions: bool = log_predictions
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        mlflow.set_tracking_uri(os.path.abspath(os.curdir) + "/mlruns")
        mlflow.set_experiment(experiment_name)

    def __enter__(self) -> "SentimentExperiment":
        self.start_time = time.time()
        mlflow.start_run()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        mlflow.log_metric("total_time_seconds", elapsed)
        mlflow.end_run()

    def evaluate(self, texts: List[str], user_ids: Optional[List[str]] = None) -> None:
        """
        Run predictions and log results to MLflow.

        Parameters:
            texts (List[str]): List of input texts.
            user_ids (Optional[List[str]]): User IDs for user-level aggregation.
        """
        predictions = self.analyzer.predict(texts)
        mlflow.log_param("model_name", self.analyzer.model_name)
        if hasattr(self.analyzer, "mode"):
            mlflow.log_param("mode", self.analyzer.mode)
        if hasattr(self.analyzer, "class_labels"):
            mlflow.log_param("class_labels", json.dumps(self.analyzer.class_labels))
        if hasattr(self.analyzer, "hypothesis_template"):
            mlflow.log_param(
                "hypothesis_template", json.dumps(self.analyzer.hypothesis_template)
            )
        if self.log_predictions:
            run_id = mlflow.active_run().info.run_id
            preds_file = f"predictions_{run_id}.txt"
            with open(preds_file, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(
                        f"Text: {p['text']}\nPredicted: {p['predicted_sentiment']}\n----\n"
                    )
            mlflow.log_artifact(preds_file)
        if self.true_labels:
            predicted_labels = [p["predicted_sentiment"] for p in predictions]
            self.log_metrics(self.true_labels, predicted_labels, user_ids)

    def log_metrics(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        user_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Compute and log classification metrics and confusion matrix.

        Parameters:
            true_labels (List[str]): True sentiment labels.
            predicted_labels (List[str]): Predicted sentiment labels.
            user_ids (Optional[List[str]]): User IDs for additional metrics.
        """
        run_id = mlflow.active_run().info.run_id
        report_str = classification_report(true_labels, predicted_labels)
        report_file = f"classification_report_{run_id}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_str)
        mlflow.log_artifact(report_file)
        report_dict = classification_report(
            true_labels, predicted_labels, output_dict=True
        )
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report_dict["macro avg"]["f1-score"])
        if "accuracy" in report_dict:
            mlflow.log_metric("accuracy", report_dict["accuracy"])
        neg_scores = report_dict.get("negative")
        if neg_scores:
            mlflow.log_metric("precision_negative", neg_scores["precision"])
            mlflow.log_metric("recall_negative", neg_scores["recall"])
            mlflow.log_metric("f1_negative", neg_scores["f1-score"])
        if user_ids:
            user_level_metrics = calculate_user_level_metrics(
                user_ids, predicted_labels, true_labels
            )
            mlflow.log_metric(
                "precision_user_macro", user_level_metrics["macro avg"]["precision"]
            )
            mlflow.log_metric(
                "recall_user_macro", user_level_metrics["macro avg"]["recall"]
            )
            mlflow.log_metric(
                "f1_user_macro", user_level_metrics["macro avg"]["f1-score"]
            )
            neg_user_scores = user_level_metrics.get("negative")
            mlflow.log_metric("precision_user_negative", neg_user_scores["precision"])
            mlflow.log_metric("recall_user_negative", neg_user_scores["recall"])
            mlflow.log_metric("f1_user_negative", neg_user_scores["f1-score"])
        exclude_keys = {"accuracy", "macro avg", "weighted avg"}
        labels = [k for k in report_dict.keys() if k not in exclude_keys]
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        cm_png = f"confusion_matrix_{run_id}.png"
        fig.savefig(cm_png, bbox_inches="tight")
        mlflow.log_artifact(cm_png)
        plt.close(fig)
