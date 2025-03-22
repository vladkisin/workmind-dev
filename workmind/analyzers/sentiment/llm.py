import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.cuda import empty_cache
from typing import List, Dict, Any, Optional
from workmind.analyzers.constants import BaseSentiment
from workmind.analyzers.sentiment.base import SentimentAnalyzerBase

DEFAULT_SYSTEM_CONTEXT: str = (
    "You are an expert in sentiment analysis. Your task is to classify the sentiment "
    "of the provided email text into one of the following categories: positive, negative, or neutral.\n"
    "Respond with only one word in lowercase indicating the sentiment. "
)

DEFAULT_USER_CONTEXT_TEMPLATE: str = (
    "Respond strictly with one word in lowercase indicating the sentiment of the provided email.\n"
    "Email text:\n{}"
)


class LLMSentimentAnalyzer(SentimentAnalyzerBase):
    """
    An LLM-based sentiment analyzer that uses a causal language model.

    It constructs a prompt from system and user contexts and then decodes the generated output.
    """

    def __init__(
        self,
        model_name: str,
        class_labels: Optional[List[str]] = None,
        batch_size: int = 16,
        bits: int = 16,
        max_input_tokens: int = 2048,
        system_context: str = DEFAULT_SYSTEM_CONTEXT,
        user_context_template: str = DEFAULT_USER_CONTEXT_TEMPLATE,
    ) -> None:
        """
        Initialize the LLM-based sentiment analyzer.

        Parameters:
            model_name (str): Hugging Face model name or path.
            class_labels (Optional[List[str]]): List of sentiment labels.
            batch_size (int): Batch size for inference.
            bits (int): Bit precision (one of 4, 8, 16, or 32).
            max_input_tokens (int): Maximum sequence length for tokenization.
            system_context (str): System prompt for the model.
            user_context_template (str): Template for the user prompt.
        """
        super().__init__(model_name, class_labels, batch_size)
        self.bits: int = bits
        self.class_labels = class_labels
        self.max_input_tokens: int = max_input_tokens
        self.system_context: str = system_context
        self.user_context_template: str = user_context_template
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """
        Load the model and tokenizer from Hugging Face with the specified bit precision.
        """
        if self.bits not in [4, 8, 16, 32]:
            raise ValueError("bits must be one of [4, 8, 16, 32].")
        bnb_config: Optional[BitsAndBytesConfig] = None
        if self.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif self.bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif self.bits == 16:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
            )
            self.model.to(self.device)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _extract_label(self, text: str) -> str:
        """
        Extract the sentiment label from the generated text.

        Parameters:
            text (str): The generated text.

        Returns:
            str: The sentiment label.
        """
        for label in self.class_labels or []:
            if label in text.lower():
                return label
        return BaseSentiment.NEUTRAL

    def _run_inference(self, text: str) -> str:
        """
        Run inference on a single text and extract the predicted sentiment.

        Parameters:
            text (str): The input text.

        Returns:
            str: The predicted sentiment.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_context},
            {"role": "user", "content": self.user_context_template.format(text)},
        ]
        tokenized_input = None
        generated_ids = None
        try:
            prompt_text: str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokenized_input = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **tokenized_input, max_new_tokens=20, use_cache=True
                )
            prompt_len: int = tokenized_input.input_ids.shape[1]
            generated_only = generated_ids[:, prompt_len:]
            response_text: str = self.tokenizer.decode(
                generated_only[0], skip_special_tokens=True
            )
        finally:
            if tokenized_input is not None:
                del tokenized_input
            if generated_ids is not None:
                del generated_ids
            empty_cache()
        return self._extract_label(response_text.strip())

    def infer_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of texts and return the predictions.

        Parameters:
            batch (List[str]): List of input texts.

        Returns:
            List[Dict[str, Any]]: List of predictions with keys 'text' and 'predicted_sentiment'.
        """
        predictions: List[Dict[str, Any]] = []
        for text in batch:
            raw_output: str = self._run_inference(text)
            predicted_sentiment: str = raw_output.lower().strip()
            predictions.append(
                {"text": text, "predicted_sentiment": predicted_sentiment}
            )
        return predictions
