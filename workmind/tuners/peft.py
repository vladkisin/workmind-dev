from typing import Dict

from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig

from workmind.tuners.base import AbstractFineTuner


class LoraCausalFineTuner(AbstractFineTuner):
    def __init__(
        self,
        model_name_or_path: str,
        train_dataset,
        eval_dataset,
        tokenizer=None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        r: int = 64,
        lora_target_modules: list[str] = None,
        bias: str = "none",
        learning_rate: float = 2e-5,
        num_train_epochs: int = 4,
        output_dir: str = "./causal_lora_output",
        compute_metrics=None,
        use_4bit: bool = True,
        train_batch_size: int = 8,
        eval_batch_size: int = 32,
    ):
        self.model_name_or_path = model_name_or_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.r = r
        self.bias = bias
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.output_dir = output_dir
        self.compute_metrics_fn = compute_metrics
        self.use_4bit = use_4bit
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.model = None
        self.trainer = None

    def prepare_model(self) -> None:
        """
        Prepares the model for LoRA fine-tuning, with optional 4-bit quantization via BitsAndBytes.
        """
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="auto",
                torch_dtype="float16",
                quantization_config=bnb_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="auto",
                torch_dtype="float16",
            )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def train(self, trainer_class=SFTTrainer) -> None:
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias="lora_only",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=1,
            learning_rate=self.learning_rate,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=False,
            lr_scheduler_type="cosine",
            eval_steps=20,
            evaluation_strategy="steps",
        )

        self.trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
        )
        self.trainer.train()

    def evaluate(self) -> Dict[str, float]:
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.evaluate()
