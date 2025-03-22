from typing import Dict, Optional, List, Any
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from workmind.tuners.base import AbstractFineTuner

DEFAULT_LORA_TARGET_MODULES: List[str] = ["q_proj", "v_proj"]


class LoraCausalFineTuner(AbstractFineTuner):
    """
    Fine-tuner using LoRA for causal language models.
    """

    def __init__(
        self,
        model_name_or_path: str,
        train_dataset: Any,
        eval_dataset: Any,
        tokenizer: Optional[Any] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        r: int = 64,
        lora_target_modules: Optional[List[str]] = None,
        bias: str = "none",
        learning_rate: float = 2e-5,
        num_train_epochs: int = 4,
        output_dir: str = "./causal_lora_output",
        compute_metrics: Optional[Any] = None,
        use_4bit: bool = True,
        train_batch_size: int = 8,
        eval_batch_size: int = 32,
    ) -> None:
        """
        Initialize the LoRA causal fine-tuner.

        Parameters:
            model_name_or_path (str): Model identifier.
            train_dataset (Any): Training dataset.
            eval_dataset (Any): Evaluation dataset.
            tokenizer (Optional[Any]): Tokenizer instance; if None, it is loaded.
            lora_alpha (int): LoRA alpha parameter.
            lora_dropout (float): LoRA dropout rate.
            r (int): LoRA rank.
            lora_target_modules (Optional[List[str]]): List of modules to target.
            bias (str): Bias setting.
            learning_rate (float): Learning rate.
            num_train_epochs (int): Number of training epochs.
            output_dir (str): Directory for output.
            compute_metrics (Optional[Any]): Metrics function.
            use_4bit (bool): Whether to use 4-bit quantization.
            train_batch_size (int): Training batch size.
            eval_batch_size (int): Evaluation batch size.
        """
        self.model_name_or_path: str = model_name_or_path
        self.train_dataset: Any = train_dataset
        self.eval_dataset: Any = eval_dataset
        self.tokenizer = tokenizer
        self.lora_alpha: int = lora_alpha
        self.lora_dropout: float = lora_dropout
        self.lora_target_modules: List[str] = (
            lora_target_modules
            if lora_target_modules is not None
            else DEFAULT_LORA_TARGET_MODULES
        )
        self.r: int = r
        self.bias: str = bias
        self.learning_rate: float = learning_rate
        self.num_train_epochs: int = num_train_epochs
        self.output_dir: str = output_dir
        self.compute_metrics_fn = compute_metrics
        self.use_4bit: bool = use_4bit
        self.train_batch_size: int = train_batch_size
        self.eval_batch_size: int = eval_batch_size
        self.model: Optional[Any] = None
        self.trainer: Optional[Any] = None

    def prepare_model(self) -> None:
        """
        Prepare the model for LoRA fine-tuning, optionally using 4-bit quantization.
        """
        if self.use_4bit:
            bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
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

    def train(self, trainer_class: Any = SFTTrainer) -> None:
        """
        Fine-tune the model using LoRA and the specified trainer.
        """
        peft_config: LoraConfig = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias="lora_only",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )
        training_args: TrainingArguments = TrainingArguments(
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
        """
        Evaluate the fine-tuned model.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.evaluate()
