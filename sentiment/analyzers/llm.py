import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from torch.cuda import empty_cache

from sentiment.constants import BaseSentiment
from sentiment.analyzers.base import SentimentAnalyzerBase


class LLMSentimentAnalyzer(SentimentAnalyzerBase):
    """
    A sentiment analyzer that uses a Causal LM model (e.g., from Hugging Face)
    to classify text sentiment as positive, negative, or neutral.
    """

    def __init__(
        self,
        model_name: str,
        class_labels=None,
        batch_size=16,
        bits: int = 16,
        max_input_tokens: int = 2048,
        system_context: str = (
            "You are an expert in sentiment analysis. Your task is to classify the sentiment "
            "of the provided email text into one of the following categories: "
            "positive, negative, or neutral.\n"
            "Respond with only one word in lowercase indicating the sentiment. "
        ),
        user_context_template: str = (
            "Respond strictly with one word in lowercase indicating the sentiment of the provided email.\n"
            "Email text:\n{}"
        ),
    ):
        """
        :param model_name: Hugging Face model name or path (e.g., "Qwen/Qwen2.5-3B-Instruct").
        :param class_labels: Optional list of class labels (e.g., ["negative", "neutral", "positive"]).
        :param batch_size: Batch size for inference.
        :param bits: Precision mode. Must be one of 32, 16, 8, or 4.
        :param max_input_tokens: Maximum sequence length for tokenizing each text.
        :param system_context: System prompt to steer the LLM's behavior.
        :param user_context_template: User prompt template for classification,
                                      containing a placeholder `{}` for the email text.
        """
        super().__init__(model_name, class_labels, batch_size)
        self.bits = bits
        self.class_labels = class_labels
        self.max_input_tokens = max_input_tokens
        self.system_context = system_context
        self.user_context_template = user_context_template

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Load model/tokenizer from Hugging Face with the specified bit precision.
        """
        if self.bits not in [4, 8, 16, 32]:
            raise ValueError("bits must be one of [4, 8, 16, 32].")

        print(f"Loading model '{self.model_name}' with {self.bits}-bit precision...")

        bnb_config = None
        if self.bits == 4:
            # 4-bit via bitsandbytes
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
                torch_dtype=torch.float16
            )
        elif self.bits == 8:
            # 8-bit via bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        elif self.bits == 16:
            # FP16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:  # self.bits == 32
            # FP32
            # You can still set device_map="auto" if your system has enough memory,
            # or you can do a direct `to(self.device)` if needed.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",  # optional
            )
            self.model.to(self.device)

        # Some models (like Qwen) require disabling cache:
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Ensure we have a pad token:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Model and tokenizer loaded successfully.")

    def _extract_label(self, text):
        for label in self.class_labels:
            if label in text.lower():
              return label
        return BaseSentiment.NEUTRAL

    def _run_inference(self, text: str) -> str:
        """
        A helper method to run inference on a single text string.
        It uses apply_chat_template to prepare the prompt.
        """
        messages = [
            {"role": "system", "content": self.system_context},
            {"role": "user", "content": self.user_context_template.format(text)},
        ]

        try:
            # Step 1: Use the tokenizer's apply_chat_template (Qwen-like approach)
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Step 2: Tokenize with truncation
            tokenized_input = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens
            ).to(self.device)

            # Step 3: Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **tokenized_input,
                    max_new_tokens=20,
                    use_cache=True
                )

            # Step 4: Remove the prompt tokens from the output to isolate the model's answer
            prompt_len = tokenized_input.input_ids.shape[1]
            generated_only = generated_ids[:, prompt_len:]

            # Decode
            response_text = self.tokenizer.decode(generated_only[0], skip_special_tokens=True)
        finally:
            # Manually clear GPU cache
            del tokenized_input
            del generated_ids
            empty_cache()

        return self._extract_label(response_text.strip())

    def infer_batch(self, batch):
        """
        Runs inference on each text in the batch and returns a list of sentiment predictions.
        Return structure: [{"text": ..., "predicted_sentiment": ...}, ...]
        """
        predictions = []
        for text in batch:
            raw_output = self._run_inference(text)
            # The system/user prompts are designed to produce a single word in lowercase.
            # We'll normalize just in case:
            predicted_sentiment = raw_output.lower().strip()

            predictions.append({
                "text": text,
                "predicted_sentiment": predicted_sentiment
            })
        return predictions
