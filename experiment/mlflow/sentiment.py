import time
import json
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from experiment.utils import calculate_user_level_metrics


class SentimentExperiment:
    """
    Class to log and evaluate sentiment analysis experiments using MLflow.
    """

    def __init__(self, analyzer, experiment_name, true_labels=None, log_predictions=False):
        """
        :param analyzer: A SentimentAnalyzerBase (or similar) object with a `predict(texts)` method
        :param experiment_name: Name of the MLflow experiment
        :param true_labels: Optional list of ground-truth labels for evaluation
        :param log_predictions: Whether to log all predictions as a text artifact. Defaults to False.
        """
        self.analyzer = analyzer
        self.experiment_name = experiment_name
        self.true_labels = true_labels
        self.log_predictions = log_predictions
        self.start_time = None
        self.end_time = None

        # Ensure experiment is created or set as current
        mlflow.set_tracking_uri(os.path.abspath(os.curdir) + '/mlruns')
        mlflow.set_experiment(experiment_name)

    def __enter__(self):
        """Start the MLflow run and timer."""
        self.start_time = time.time()
        mlflow.start_run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End the MLflow run and log total time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        mlflow.log_metric("total_time_seconds", elapsed)
        mlflow.end_run()

    def evaluate(self, texts, user_ids=None):
        """
        Run predictions and log results to MLflow.
        - Optionally log predictions (as a text artifact) if log_predictions=True
        - If true_labels is provided, log classification metrics and a confusion matrix plot
        """
        # Generate predictions
        predictions = self.analyzer.predict(texts)

        # Log model-related parameters
        mlflow.log_param("model_name", self.analyzer.model_name)
        # If "mode" is not defined in all analyzers, wrap in a hasattr check
        if hasattr(self.analyzer, "mode"):
            mlflow.log_param("mode", self.analyzer.mode)

        # If the analyzer has class_labels, log them (as JSON for clarity)
        if hasattr(self.analyzer, "class_labels"):
            mlflow.log_param("class_labels", json.dumps(self.analyzer.class_labels))

        if hasattr(self.analyzer, "hypothesis_template"):
            mlflow.log_param("hypothesis_template", json.dumps(self.analyzer.class_labels))

        # Optionally log predictions as a text artifact.
        if self.log_predictions:
            run_id = mlflow.active_run().info.run_id
            preds_file = f"predictions_{run_id}.txt"
            with open(preds_file, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(f"Text: {p['text']}\n")
                    f.write(f"Predicted: {p['predicted_sentiment']}\n")
                    f.write("----\n")
            mlflow.log_artifact(preds_file)

        # If we have true labels, compute and log additional metrics
        if self.true_labels:
            predicted_labels = [p["predicted_sentiment"] for p in predictions]
            self.log_metrics(self.true_labels, predicted_labels, user_ids)

    def log_metrics(self, true_labels, predicted_labels, user_ids=None):
        """
        Compute and log classification metrics:
          - classification report (as text)
          - confusion matrix (as a PNG image)
          - macro avg precision, recall, f1
          - accuracy
          - additional metrics for 'negative' class
        """
        run_id = mlflow.active_run().info.run_id

        # 1) classification_report as text
        report_str = classification_report(true_labels, predicted_labels)
        report_file = f"classification_report_{run_id}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_str)
        mlflow.log_artifact(report_file)

        # 2) classification_report as a dict (for logging metrics)
        report_dict = classification_report(true_labels, predicted_labels, output_dict=True)

        # Log macro avg metrics
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report_dict["macro avg"]["f1-score"])

        # Also log accuracy if available
        if "accuracy" in report_dict:
            mlflow.log_metric("accuracy", report_dict["accuracy"])

        # ---------------------------------------------
        # 3) Metrics for the "negative" class specifically
        # ---------------------------------------------
        # Check if "negative" is indeed in the report_dict
        # (It should be if "negative" is a true or predicted label.)
        neg_scores = report_dict.get("negative")
        if neg_scores:
            mlflow.log_metric("precision_negative", neg_scores["precision"])
            mlflow.log_metric("recall_negative", neg_scores["recall"])
            mlflow.log_metric("f1_negative", neg_scores["f1-score"])

        if user_ids:
            user_level_metrics = calculate_user_level_metrics(user_ids, predicted_labels, true_labels)
            mlflow.log_metric("precision_user_macro", user_level_metrics["macro avg"]["precision"])
            mlflow.log_metric("recall_user_macro", user_level_metrics["macro avg"]["recall"])
            mlflow.log_metric("f1_user_macro", user_level_metrics["macro avg"]["f1-score"])
            neg_user_scores = user_level_metrics.get("negative")
            mlflow.log_metric("precision_user_negative", neg_user_scores["precision"])
            mlflow.log_metric("recall_user_negative", neg_user_scores["recall"])
            mlflow.log_metric("f1_user_negative", neg_user_scores["f1-score"])

        # 4) Confusion matrix with label axis
        # Create a list of labels from the report_dict keys (excluding avg keys)
        # so the heatmap has consistent ordering for x and y.
        exclude_keys = {"accuracy", "macro avg", "weighted avg"}
        labels = [k for k in report_dict.keys() if k not in exclude_keys]

        # Generate confusion matrix with specific label order
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        cm_png = f"confusion_matrix_{run_id}.png"
        fig.savefig(cm_png, bbox_inches='tight')
        mlflow.log_artifact(cm_png)
        plt.close(fig)
