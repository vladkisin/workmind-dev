from typing import Dict
from trl import SFTTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from workmind.tuners.base import AbstractFineTuner
from workmind.tuners.utils import default_compute_metrics


class PartiallyUnfrozenClsFineTuner(AbstractFineTuner):
    def __init__(
        self,
        model_name_or_path: str,
        train_dataset,
        eval_dataset,
        tokenizer,
        layers_to_unfreeze=("layer.21", "layer.22", "layer.23", "classifier"),
        compute_metrics=default_compute_metrics,
        output_dir: str = "./results",
        learning_rate: float = 1e-5,
        num_train_epochs: int = 15,
        num_labels=3,
        train_batch_size=16,
        val_batch_size=16,
    ):
        self.model_name_or_path = model_name_or_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.layers_to_unfreeze = layers_to_unfreeze
        self.compute_metrics_fn = compute_metrics
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.num_labels = num_labels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.model = None
        self.trainer = None

    def prepare_model(self) -> None:
        config = AutoConfig.from_pretrained(
            self.model_name_or_path, num_labels=self.num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=config
        )

        for name, param in self.model.named_parameters():
            if any(layer_id in name for layer_id in self.layers_to_unfreeze):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train(self, trainer_class=SFTTrainer) -> None:
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
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.evaluate()
