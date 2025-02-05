import time
import wandb
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from experiment.utils import compute_bertscore, compute_cosine_similarity, compute_bleu, compute_rouge, \
    compute_perplexity, compute_readability


class InterventionExperiment:
    def __init__(self, intervention_generator, experiment_name, anchor_model=None,
                 project_name="workmind-interventions", log_predictions=False, batch_size=8):
        """
        :param intervention_generator: A model object with a predict(texts) method for generating interventions.
        :param anchor_model: A model object with a predict(texts) method (serving as the pseudo–ground truth).
        :param experiment_name: Name for the W&B run.
        :param project_name: W&B project name.
        :param log_predictions: If True, predictions are logged as a text artifact.
        :param batch_size: Batch size used for predictions and encoding.
        """
        self.intervention_generator = intervention_generator
        self.anchor_model = anchor_model
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.log_predictions = log_predictions
        self.batch_size = batch_size
        self.run = None
        self.start_time = None
        self.end_time = None

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # TODO: refactor to use constants or parametrize
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.lm_model = AutoModelForCausalLM.from_pretrained("gpt2")

    def __enter__(self):
        """Start the W&B run and timer."""
        self.start_time = time.time()
        self.run = wandb.init(project=self.project_name, name=self.experiment_name, reinit=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Log total time and finish the W&B run."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        wandb.log({"total_time_seconds": elapsed})
        self.run.finish()

    def calculate_metrics(self, candidate_outputs, anchor_outputs):
        """
        Calculate evaluation metrics for two sets of generated texts.

        :param candidate_outputs: List of texts from the candidate model.
        :param anchor_outputs: List of texts from the anchor model.
        :return: A tuple (metrics, df) where metrics is a dict of aggregated metrics,
                 and df is a detailed pandas DataFrame.
        """
        bert_scores = compute_bertscore(candidate_outputs, anchor_outputs)
        cosine_sim = compute_cosine_similarity(candidate_outputs, anchor_outputs, self.embedding_model,
                                               batch_size=self.batch_size)
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
            "readability": readability_scores
        })

        metrics = {
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
            "readability_distribution": readability_scores
        }

        return metrics, df

    def log_metrics(self, anchor_outputs, candidate_outputs):
        metrics, df = self.calculate_metrics(candidate_outputs, anchor_outputs)
        wandb.log({
            "avg_bertscore_f1": metrics["avg_bertscore_f1"],
            "avg_cosine_similarity": metrics["avg_cosine_similarity"],
            "avg_bleu": metrics["avg_bleu"],
            "rougeL": metrics["rougeL"],
            "avg_perplexity": metrics["avg_perplexity"],
            "avg_readability": metrics["avg_readability"]
        })
        wandb.log({"evaluation_table": wandb.Table(dataframe=df)})

    def evaluate(self, input_texts, anchor_outputs):
        predictions = []
        for texts in tqdm(input_texts):
            result = self.intervention_generator.analyze_emails([texts])
            predictions.append(result)
        self.log_metrics(predictions, anchor_outputs)
