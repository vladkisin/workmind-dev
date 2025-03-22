from typing import Dict, Optional, Any
from adapters import AutoAdapterModel, AdapterTrainer
from transformers import AutoConfig, TrainingArguments
from workmind.tuners.base import AbstractFineTuner
from workmind.tuners.utils import default_compute_metrics

ADAPTER_CONFIG: str = "lora"


class AdapterFineTuner(AbstractFineTuner):
    """
    Fine-tuner for adapter-based models.
    """

    def __init__(
        self,
        model_name_or_path: str,
        train_dataset: Any,
        eval_dataset: Any,
        tokenizer: Any,
        adapter_name: str = "sentiment-head",
        num_labels: int = 2,
        id2label: Optional[Dict[int, str]] = None,
        compute_metrics: Any = default_compute_metrics,
        output_dir: str = "./training_output",
        learning_rate: float = 5e-5,
        num_train_epochs: int = 2,
        train_batch_size: int = 8,
        eval_batch_size: int = 32,
        eval_steps: int = 10,
    ) -> None:
        """
        Initialize the adapter fine-tuner.

        Parameters:
            model_name_or_path (str): Model identifier.
            train_dataset (Any): Training dataset.
            eval_dataset (Any): Evaluation dataset.
            tokenizer (Any): Tokenizer instance.
            adapter_name (str): Name of the adapter.
            num_labels (int): Number of labels.
            id2label (Optional[Dict[int, str]]): Mapping from id to label.
            compute_metrics (Any): Metrics function.
            output_dir (str): Output directory.
            learning_rate (float): Learning rate.
            num_train_epochs (int): Number of epochs.
            train_batch_size (int): Training batch size.
            eval_batch_size (int): Evaluation batch size.
            eval_steps (int): Evaluation logging steps.
        """
        self.model_name_or_path: str = model_name_or_path
        self.train_dataset: Any = train_dataset
        self.eval_dataset: Any = eval_dataset
        self.adapter_name: str = adapter_name
        self.num_labels: int = num_labels
        self.id2label: Optional[Dict[int, str]] = id2label
        self.compute_metrics_fn = compute_metrics
        self.output_dir: str = output_dir
        self.learning_rate: float = learning_rate
        self.num_train_epochs: int = num_train_epochs
        self.train_batch_size: int = train_batch_size
        self.eval_batch_size: int = eval_batch_size
        self.eval_steps: int = eval_steps
        self.model: Optional[Any] = None
        self.trainer: Optional[Any] = None

    def prepare_model(self) -> None:
        """
        Load the model from a pretrained checkpoint, add the adapter and classification head,
        and activate the adapter.
        """
        config = AutoConfig.from_pretrained(
            self.model_name_or_path, num_labels=self.num_labels
        )
        self.model = AutoAdapterModel.from_pretrained(
            self.model_name_or_path, config=config
        )
        self.model.add_adapter(self.adapter_name, config=ADAPTER_CONFIG)
        self.model.add_classification_head(
            self.adapter_name, num_labels=self.num_labels, id2label=self.id2label
        )
        self.model.train_adapter(self.adapter_name)

    def train(self, trainer_class: Any = AdapterTrainer) -> None:
        """
        Train the model using the specified trainer class.

        Parameters:
            trainer_class (Any): Trainer class to use.
        """
        training_args = TrainingArguments(
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            logging_dir="./logs",
            logging_steps=self.eval_steps,
            logging_first_step=True,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            eval_strategy="steps",
            warmup_steps=1,
            lr_scheduler_type="linear",
            report_to="none",
            save_steps=100,
            evaluation_strategy="steps",
        )
        self.trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
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
