from typing import Dict, Optional

from adapters import AutoAdapterModel, AdapterTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from tuners.base import AbstractFineTuner
from tuners.utils import default_compute_metrics


class AdapterFineTuner(AbstractFineTuner):
    def __init__(
            self,
            model_name_or_path: str,
            train_dataset,
            eval_dataset,
            tokenizer,
            adapter_name: str = "sentiment-head",
            num_labels: int = 2,
            id2label: Optional[Dict[int, str]] = None,
            compute_metrics=default_compute_metrics,
            output_dir: str = "./training_output",
            learning_rate: float = 5e-5,
            num_train_epochs: int = 2,
            train_batch_size: int = 8,
            eval_batch_size: int = 32,
            eval_steps: int = 10
    ):
        self.model_name_or_path = model_name_or_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.adapter_name = adapter_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.compute_metrics_fn = compute_metrics
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.model = None
        self.trainer = None

    def prepare_model(self) -> None:
        config = AutoConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoAdapterModel.from_pretrained(self.model_name_or_path, config=config)

        self.model.add_adapter(self.adapter_name, config="lora")  # or other adapter config if you prefer

        self.model.add_classification_head(
            self.adapter_name,
            num_labels=self.num_labels,
            id2label=self.id2label
        )

        # Activate the adapter
        self.model.train_adapter(self.adapter_name)

    def train(self, trainer_class=AdapterTrainer) -> None:
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
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.evaluate()
