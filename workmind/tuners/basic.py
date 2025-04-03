from typing import Dict, Any, Optional
from trl import SFTTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from workmind.tuners.base import AbstractFineTuner
from workmind.tuners.utils import default_compute_metrics


class BaseClsFineTuner(AbstractFineTuner):
    """
    Base fine-tuner class that implements common logic for training and evaluation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        train_dataset: Any,
        eval_dataset: Any,
        tokenizer: Any,
        compute_metrics: Any = default_compute_metrics,
        output_dir: str = "./results",
        learning_rate: float = 1e-5,
        num_train_epochs: int = 15,
        num_labels: int = 3,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
    ) -> None:
        """
        Initialize the fine-tuner.

        Parameters:
            model_name_or_path (str): Model identifier.
            train_dataset (Any): Training dataset.
            eval_dataset (Any): Evaluation dataset.
            tokenizer (Any): Tokenizer instance.
            compute_metrics (Any): Function to compute metrics.
            output_dir (str): Output directory.
            learning_rate (float): Learning rate.
            num_train_epochs (int): Number of training epochs.
            num_labels (int): Number of output classes.
            train_batch_size (int): Training batch size.
            val_batch_size (int): Validation batch size.
        """
        self.model_name_or_path = model_name_or_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics_fn = compute_metrics
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.num_labels = num_labels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.model: Optional[Any] = None
        self.trainer: Optional[Any] = None

    def prepare_model(self) -> None:
        """
        Load the model. Subclasses should override this method to modify parameter freezing.
        """
        config = AutoConfig.from_pretrained(
            self.model_name_or_path, num_labels=self.num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=config
        )

    def train(self, trainer_class: Any = SFTTrainer) -> None:
        """
        Train the model using the specified trainer.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.val_batch_size,
            num_train_epochs=self.num_train_epochs,
            report_to="none",
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            save_total_limit=0,
        )
        self.trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_fn,
        )
        self.trainer.train()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the fine-tuned model.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.evaluate()


class FullyUnfrozenClsFineTuner(BaseClsFineTuner):
    """
    Fine-tuner that fine-tunes the whole model (all layers are trainable).
    """

    def prepare_model(self) -> None:
        """
        Load the model and ensure that all layers are trainable.
        """
        super().prepare_model()
        for name, param in self.model.named_parameters():
            param.requires_grad = True
