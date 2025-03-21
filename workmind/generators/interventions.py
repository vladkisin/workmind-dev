import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from workmind.generators.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    get_entity_user_prompt,
    get_entity_system_prompt,
)


class InterventionGenerator:
    def __init__(
        self,
        model_name="microsoft/Phi-3.5-mini-instruct",
        max_input_tokens=512,
        max_output_tokens=512,
        batch_size=1,
        load_in_8bit=False,
        load_in_4bit=False,
        entity="email(s)",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_USER_PROMPT,
    ):
        """
        Initializes the InterventionGenerator with model, tokenizer, and batching settings.
        """
        self.max_input_tokens = max_input_tokens
        self.batch_size = batch_size
        self.max_output_tokens = max_output_tokens
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

        self.system_context = get_entity_system_prompt(system_prompt, entity)
        self.user_context_template = get_entity_user_prompt(user_prompt, entity)

    def construct_prompt(self, emails):
        """
        Constructs a formatted prompt for the model.
        """
        return [
            {"role": "system", "content": self.system_context},
            {
                "role": "user",
                "content": self.user_context_template.format(
                    "\n".join(f"{i + 1}. {text}" for i, text in enumerate(emails))
                ),
            },
        ]

    def preprocess_emails(self, batch_of_emails):
        """
        Prepares and tokenizes a batch of emails for the model.
        """
        messages = [self.construct_prompt(emails) for emails in batch_of_emails]
        texts = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        # Tokenize as a batch
        tokenized_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        return tokenized_inputs

    def analyze_emails(self, batch_of_emails):
        """
        Analyzes a batch of emails for dissatisfaction and generates structured interventions.
        """
        self.logger.info(f"Processing batch of {len(batch_of_emails)} emails...")

        try:
            tokenized_inputs = self.preprocess_emails(batch_of_emails)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **tokenized_inputs,
                    max_new_tokens=self.max_output_tokens,
                    use_cache=True,
                )

            response_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(
                    tokenized_inputs.input_ids, generated_ids
                )
            ]
            responses = self.tokenizer.batch_decode(
                response_ids, skip_special_tokens=True
            )

            for i, response in enumerate(responses):
                self.logger.info(f"Email {i + 1} response: {response.strip()}")

            return responses

        except Exception as e:
            self.logger.error(f"Error during batch analysis: {e}", exc_info=True)
            raise e

        finally:
            # Cleanup GPU memory
            del tokenized_inputs
            del generated_ids
            torch.cuda.empty_cache()

    def predict(self, emails):
        """
        Processes emails in batches based on the specified batch size.
        """
        results = []
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
    A Retrieval-Augmented Generator for interventions based on dissatisfaction detection.
    Uses llama-index for retrieving context from a given index and an LLM to generate responses.
    """

    def __init__(
        self,
        llm,
        index,
        text_qa_template,
        refine_template,
        batch_size=1,
    ):
        """
        Initializes the RAGInterventionGenerator with:
        - An LLM (HuggingFaceLLM from llama-index or compatible).
        - A llama-index instance (VectorStoreIndex or similar).
        - System and user prompts.
        - Llama-index ChatPromptTemplate instances for initial query and refinement.
        - Batch size for processing emails.
        """
        self.llm = llm
        self.index = index
        self.text_qa_template = text_qa_template
        self.refine_template = refine_template
        self.batch_size = batch_size

        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def construct_prompt(emails):
        """
        Constructs a formatted query from the given emails to be passed to the llama-index query engine.
        This is analogous to building a prompt in a standard LLM scenario.
        """
        email_list_str = "\n".join(f"{i+1}. {text}" for i, text in enumerate(emails))
        return email_list_str

    def preprocess_emails(self, batch_of_emails):
        return [self.construct_prompt(emails) for emails in batch_of_emails]

    def analyze_emails(self, batch_of_emails):
        self.logger.info(f"Processing batch of {len(batch_of_emails)} emails...")

        try:
            prompts = self.preprocess_emails(batch_of_emails)
            responses = []

            for prompt_text in prompts:
                query_result = self.query_engine.query(prompt_text)
                response_text = query_result.response
                responses.append(response_text)
            return responses

        except Exception as e:
            self.logger.error(f"Error during batch analysis: {e}", exc_info=True)
            raise e

        finally:
            torch.cuda.empty_cache()

    def predict(self, emails):
        results = []
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
