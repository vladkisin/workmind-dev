import time
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from experiment.utils import calculate_user_level_metrics


class SentimentExperiment:
    """
    Class to log and evaluate sentiment analysis experiments using W&B.
    """

    def __init__(self, analyzer, experiment_name, true_labels=None, log_predictions=False, project_name="workmind"):
        """
        :param analyzer: A SentimentAnalyzerBase (or similar) object with a `predict(texts)` method
        :param project_name: Name of the W&B project
        :param true_labels: Optional list of ground-truth labels for evaluation
        :param log_predictions: Whether to log all predictions as a text artifact. Defaults to False.
        """
        self.analyzer = analyzer
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.true_labels = true_labels
        self.log_predictions = log_predictions
        self.run = None
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start the W&B run and timer."""
        self.start_time = time.time()
        self.run = wandb.init(project=self.project_name, name=self.experiment_name, reinit=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End the W&B run and log total time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        wandb.log({"total_time_seconds": elapsed})
        self.run.finish()

    def evaluate(self, texts, user_ids=None):
        """
        Run predictions and log results to W&B.
        - Optionally log predictions (as a text artifact) if log_predictions=True
        - If true_labels is provided, log classification metrics and a confusion matrix plot
        """
        # Generate predictions
        predictions = self.analyzer.predict(texts)

        # Log model-related parameters
        wandb.config.update({"model_name": self.analyzer.model_name}, allow_val_change=True)
        if hasattr(self.analyzer, "mode"):
            wandb.config.update({"mode": self.analyzer.mode}, allow_val_change=True)
        if hasattr(self.analyzer, "class_labels"):
            wandb.config.update({"class_labels": self.analyzer.class_labels}, allow_val_change=True)
        if hasattr(self.analyzer, "hypothesis_template"):
            wandb.config.update({"hypothesis_template": self.analyzer.hypothesis_template}, allow_val_change=True)

        # Optionally log predictions as a text artifact
        if self.log_predictions:
            preds_file = "predictions.txt"
            with open(preds_file, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(f"Text: {p['text']}\n")
                    f.write(f"Predicted: {p['predicted_sentiment']}\n")
                    f.write("----\n")
            wandb.save(preds_file)

        # If we have true labels, compute and log additional metrics
        if self.true_labels:
            predicted_labels = [p["predicted_sentiment"] for p in predictions]
            self.log_metrics(self.true_labels, predicted_labels, user_ids)

    def log_metrics(self, true_labels, predicted_labels, user_ids=None):
        """
        Compute and log classification metrics:
          - classification report (as text and dict)
          - confusion matrix (as an image)
          - macro avg precision, recall, f1
          - accuracy
          - additional metrics for 'negative' class
        """
        # 1) classification_report as dict and log metrics
        report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
        wandb.log({"classification_report": report_dict})

        # Log macro avg metrics
        wandb.log({
            "precision_macro": report_dict["macro avg"]["precision"],
            "recall_macro": report_dict["macro avg"]["recall"],
            "f1_macro": report_dict["macro avg"]["f1-score"]
        })

        # Also log accuracy if available
        if "accuracy" in report_dict:
            wandb.log({"accuracy": report_dict["accuracy"]})

        # ---------------------------------------------
        # 2) Metrics for the "negative" class specifically
        # ---------------------------------------------
        neg_scores = report_dict.get("negative")
        if neg_scores:
            wandb.log({
                "precision_negative": neg_scores["precision"],
                "recall_negative": neg_scores["recall"],
                "f1_negative": neg_scores["f1-score"]
            })

        if user_ids:
            user_level_metrics = calculate_user_level_metrics(user_ids, predicted_labels, true_labels)
            wandb.log({
                "precision_user_macro": user_level_metrics["macro avg"]["precision"],
                "recall_user_macro": user_level_metrics["macro avg"]["recall"],
                "f1_user_macro": user_level_metrics["macro avg"]["f1-score"]
            })
            neg_user_scores = user_level_metrics.get("negative")
            wandb.log({
                "precision_user_negative": neg_user_scores["precision"],
                "recall_user_negative": neg_user_scores["recall"],
                "f1_user_negative": neg_user_scores["f1-score"]
            })

        # 3) Confusion matrix with label axis
        exclude_keys = {"accuracy", "macro avg", "weighted avg"}
        labels = [k for k in report_dict.keys() if k not in exclude_keys]

        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        cm_png = "confusion_matrix.png"
        fig.savefig(cm_png, bbox_inches='tight')
        wandb.log({"confusion_matrix": wandb.Image(cm_png)})
        plt.close(fig)
