import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

from workmind.generators.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    get_entity_user_prompt,
    get_entity_system_prompt,
)

DEFAULT_MAX_INPUT_TOKENS: int = 512
DEFAULT_MAX_OUTPUT_TOKENS: int = 512


class InterventionGenerator:
    """
    Generates structured HR interventions based on input email text.

    This generator uses a causal language model to generate intervention recommendations.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        batch_size: int = 1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        entity: str = "email(s)",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> None:
        """
        Initialize the InterventionGenerator.

        Parameters:
            model_name (str): Model identifier.
            max_input_tokens (int): Maximum tokens for input.
            max_output_tokens (int): Maximum tokens for generated output.
            batch_size (int): Batch size for processing.
            load_in_8bit (bool): Whether to load model in 8-bit mode.
            load_in_4bit (bool): Whether to load model in 4-bit mode.
            entity (str): Entity description used in prompts.
            system_prompt (str): System prompt template.
            user_prompt (str): User prompt template.
        """
        self.max_input_tokens: int = max_input_tokens
        self.batch_size: int = batch_size
        self.max_output_tokens: int = max_output_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.system_context: str = get_entity_system_prompt(system_prompt, entity)
        self.user_context_template: str = get_entity_user_prompt(user_prompt, entity)

    def construct_prompt(self, emails: List[str]) -> List[Dict[str, str]]:
        """
        Construct a prompt using the system and user context templates.

        Parameters:
            emails (List[str]): List of email text strings.

        Returns:
            List[Dict[str, str]]: A list of messages forming the prompt.
        """
        user_content: str = "\n".join(
            f"{i + 1}. {text}" for i, text in enumerate(emails)
        )
        return [
            {"role": "system", "content": self.system_context},
            {
                "role": "user",
                "content": self.user_context_template.format(user_content),
            },
        ]

    def preprocess_emails(self, batch_of_emails: List[List[str]]) -> Any:
        """
        Prepares and tokenizes a batch of emails for the model.

        Parameters:
            batch_of_emails (List[List[str]]): A list of email batches (each batch is a list of email strings).

        Returns:
            Any: Tokenized inputs suitable for the model.
        """
        messages: List[List[Dict[str, str]]] = [
            self.construct_prompt(emails) for emails in batch_of_emails
        ]
        texts: List[str] = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        tokenized_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        return tokenized_inputs

    def analyze_emails(self, batch_of_emails: List[List[str]]) -> List[str]:
        """
        Analyze a batch of emails and generate intervention responses.

        Parameters:
            batch_of_emails (List[List[str]]): A list of email batches.

        Returns:
            List[str]: A list of generated intervention responses.
        """
        self.logger.info(f"Processing batch of {len(batch_of_emails)} emails...")
        tokenized_inputs = None
        generated_ids = None
        try:
            tokenized_inputs = self.preprocess_emails(batch_of_emails)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **tokenized_inputs,
                    max_new_tokens=self.max_output_tokens,
                    use_cache=True,
                )
            prompt_len: int = tokenized_inputs.input_ids.shape[1]
            response_ids = [
                output_ids[prompt_len:]
                for input_ids, output_ids in zip(
                    tokenized_inputs.input_ids, generated_ids
                )
            ]
            responses: List[str] = self.tokenizer.batch_decode(
                response_ids, skip_special_tokens=True
            )
            for i, response in enumerate(responses):
                self.logger.info(f"Email {i + 1} response: {response.strip()}")
            return responses
        except Exception as e:
            self.logger.error(f"Error during batch analysis: {e}", exc_info=True)
            raise e
        finally:
            if tokenized_inputs is not None:
                del tokenized_inputs
            if generated_ids is not None:
                del generated_ids
            torch.cuda.empty_cache()

    def predict(self, emails: List[List[str]]) -> List[str]:
        """
        Process emails in batches and return generated intervention responses.

        Parameters:
            emails (List[List[str]]): List of email batches.

        Returns:
            List[str]: Generated interventions.
        """
        results: List[str] = []
        for i in range(0, len(emails), self.batch_size):
            batch = emails[i : i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1}...")
            try:
                batch_results = self.analyze_emails(batch)
                results.extend(batch_results)
            except Exception as e:
                self.logger.warning(
                    f"Skipping batch {i // self.batch_size + 1} due to error: {e}"
                )
        return results


class RAGInterventionGenerator:
    """
    Retrieval-Augmented Generator for interventions.

    This generator uses an external index and a language model to retrieve and refine context before generating responses.
    """

    def __init__(
        self,
        llm: Any,
        index: Any,
        text_qa_template: Any,
        refine_template: Any,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the RAG intervention generator.

        Parameters:
            llm (Any): A language model instance (e.g., HuggingFaceLLM from llama-index).
            index (Any): A retrieval index instance.
            text_qa_template (Any): Template for initial query.
            refine_template (Any): Template for refining the response.
            batch_size (int): Batch size for processing.
        """
        self.llm: Any = llm
        self.index: Any = index
        self.text_qa_template: Any = text_qa_template
        self.refine_template: Any = refine_template
        self.batch_size: int = batch_size
        self.query_engine: Any = self.index.as_query_engine(
            llm=self.llm,
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def construct_prompt(emails: List[str]) -> str:
        """
        Construct a prompt from the given emails.

        Parameters:
            emails (List[str]): List of email texts.

        Returns:
            str: A formatted prompt string.
        """
        return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(emails))

    def preprocess_emails(self, batch_of_emails: List[List[str]]) -> List[str]:
        """
        Prepare email batches as prompts.

        Parameters:
            batch_of_emails (List[List[str]]): List of email batches.

        Returns:
            List[str]: List of prompt strings.
        """
        return [self.construct_prompt(emails) for emails in batch_of_emails]

    def analyze_emails(self, batch_of_emails: List[List[str]]) -> List[str]:
        """
        Analyze a batch of emails using the query engine.

        Parameters:
            batch_of_emails (List[List[str]]): List of email batches.

        Returns:
            List[str]: List of generated intervention responses.
        """
        self.logger.info(f"Processing batch of {len(batch_of_emails)} emails...")
        try:
            prompts: List[str] = self.preprocess_emails(batch_of_emails)
            responses: List[str] = []
            for prompt_text in prompts:
                query_result = self.query_engine.query(prompt_text)
                responses.append(query_result.response)
            return responses
        except Exception as e:
            self.logger.error(f"Error during batch analysis: {e}", exc_info=True)
            raise e
        finally:
            torch.cuda.empty_cache()

    def predict(self, emails: List[List[str]]) -> List[str]:
        """
        Process emails in batches and return generated intervention responses.

        Parameters:
            emails (List[List[str]]): List of email batches.

        Returns:
            List[str]: Generated interventions.
        """
        results: List[str] = []
        for i in range(0, len(emails), self.batch_size):
            batch = emails[i : i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1}...")
            try:
                batch_results = self.analyze_emails(batch)
                results.extend(batch_results)
            except Exception as e:
                self.logger.warning(
                    f"Skipping batch {i // self.batch_size + 1} due to error: {e}"
                )
        return results
