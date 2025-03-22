import time
import wandb
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from workmind.experiment.utils import (
    compute_bertscore,
    compute_cosine_similarity,
    compute_bleu,
    compute_rouge,
    compute_perplexity,
    compute_readability,
)


class InterventionExperiment:
    """
    W&B-based experiment wrapper for logging and evaluating intervention generation.
    """

    def __init__(
        self,
        intervention_generator: Any,
        experiment_name: str,
        anchor_model: Optional[Any] = None,
        project_name: str = "workmind-interventions",
        log_predictions: bool = False,
        batch_size: int = 8,
    ) -> None:
        """
        Initialize the intervention experiment.

        Parameters:
            intervention_generator (Any): Model object with a predict method.
            experiment_name (str): Name for the W&B run.
            anchor_model (Optional[Any]): Model used as pseudo–ground truth.
            project_name (str): W&B project name.
            log_predictions (bool): Whether to log predictions as an artifact.
            batch_size (int): Batch size for predictions.
        """
        self.intervention_generator: Any = intervention_generator
        self.anchor_model: Optional[Any] = anchor_model
        self.experiment_name: str = experiment_name
        self.project_name: str = project_name
        self.log_predictions: bool = log_predictions
        self.batch_size: int = batch_size
        self.run: Optional[Any] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.embedding_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.lm_model = AutoModelForCausalLM.from_pretrained("gpt2")

    def __enter__(self) -> "InterventionExperiment":
        self.start_time = time.time()
        self.run = wandb.init(project=self.project_name, name=self.experiment_name, reinit=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        wandb.log({"total_time_seconds": elapsed})
        self.run.finish()

    def calculate_metrics(self, candidate_outputs: List[str], anchor_outputs: List[str]) -> (Dict[str, Any], pd.DataFrame):
        """
        Calculate evaluation metrics for generated interventions.

        Parameters:
            candidate_outputs (List[str]): Generated interventions.
            anchor_outputs (List[str]): Reference interventions.

        Returns:
            Tuple[Dict[str, Any], pd.DataFrame]: Aggregated metrics and detailed results DataFrame.
        """
        bert_scores = compute_bertscore(candidate_outputs, anchor_outputs)
        cosine_sim = compute_cosine_similarity(candidate_outputs, anchor_outputs, self.embedding_model, batch_size=self.batch_size)
        bleu_scores = compute_bleu(candidate_outputs, anchor_outputs)
        rouge_score = compute_rouge(candidate_outputs, anchor_outputs)
        perplexity_scores = compute_perplexity(candidate_outputs, self.tokenizer, self.lm_model)
        readability_scores = compute_readability(candidate_outputs)
        df = pd.DataFrame({
            "candidate_output": candidate_outputs,
            "anchor_output": anchor_outputs,
            "bertscore": bert_scores,
            "cosine_similarity": cosine_sim,
            "bleu": bleu_scores,
            "perplexity": perplexity_scores,
            "readability": readability_scores,
        })
        metrics: Dict[str, Any] = {
            "avg_bertscore_f1": sum(bert_scores) / len(bert_scores) if bert_scores else None,
            "avg_cosine_similarity": sum(cosine_sim) / len(cosine_sim) if cosine_sim else None,
            "avg_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else None,
            "rougeL": rouge_score,
            "avg_perplexity": sum(perplexity_scores) / len(perplexity_scores) if perplexity_scores else None,
            "avg_readability": sum(readability_scores) / len(readability_scores) if readability_scores else None,
            "bertscore_distribution": bert_scores,
            "cosine_similarity_distribution": cosine_sim,
            "bleu_distribution": bleu_scores,
            "perplexity_distribution": perplexity_scores,
            "readability_distribution": readability_scores,
        }
        return metrics, df

    def log_metrics(self, anchor_outputs: List[str], candidate_outputs: List[str]) -> None:
        metrics, df = self.calculate_metrics(candidate_outputs, anchor_outputs)
        wandb.log({
            "avg_bertscore_f1": metrics["avg_bertscore_f1"],
            "avg_cosine_similarity": metrics["avg_cosine_similarity"],
            "avg_bleu": metrics["avg_bleu"],
            "rougeL": metrics["rougeL"],
            "avg_perplexity": metrics["avg_perplexity"],
            "avg_readability": metrics["avg_readability"],
        })
        wandb.log({"evaluation_table": wandb.Table(dataframe=df)})

    def evaluate(self, input_texts: List[List[str]], anchor_outputs: List[str]) -> None:
        """
        Process and evaluate a batch of email inputs and log metrics to W&B.

        Parameters:
            input_texts (List[List[str]]): List of email batches.
            anchor_outputs (List[str]): Reference interventions.
        """
        predictions: List[str] = []
        for texts in tqdm(input_texts):
            result = self.intervention_generator.analyze_emails([texts])[0]
            predictions.append(result)
        self.log_metrics(anchor_outputs, predictions)
