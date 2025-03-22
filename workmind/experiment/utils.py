import re
import pandas as pd
import readability
import torch
from typing import List, Dict, Any
from sklearn.metrics import classification_report
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
from sentence_transformers import SentenceTransformer, util


def calculate_user_level_metrics(
    user_ids: List[str], predicted_sentiments: List[str], real_sentiments: List[str]
) -> Dict[str, Any]:
    """
    Aggregate email sentiments by user and compute classification metrics.

    Parameters:
        user_ids (List[str]): List of user IDs.
        predicted_sentiments (List[str]): Predicted sentiment labels.
        real_sentiments (List[str]): True sentiment labels.

    Returns:
        Dict[str, Any]: Classification report for user-level predictions.
    """
    data = pd.DataFrame(
        {
            "user_id": user_ids,
            "predicted": predicted_sentiments,
            "real": real_sentiments,
        }
    )
    user_level = (
        data.groupby("user_id")
        .agg(
            {
                "predicted": lambda x: (
                    "negative" if "negative" in x.values else "positive"
                ),
                "real": lambda x: "negative" if "negative" in x.values else "positive",
            }
        )
        .reset_index()
    )
    report = classification_report(
        user_level["real"], user_level["predicted"], output_dict=True
    )
    return report


def clean_text(text: str) -> str:
    """
    Clean a text string by reducing whitespace.

    Parameters:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def compute_bertscore(
    candidate_outputs: List[str],
    reference_outputs: List[str],
    model_type: str = "roberta-base",
    batch_size: int = 4,
) -> List[float]:
    """
    Compute BERTScore F1 values for candidate texts against reference texts.

    Parameters:
        candidate_outputs (List[str]): Candidate texts.
        reference_outputs (List[str]): Reference texts.
        model_type (str): Model type to use.
        batch_size (int): Batch size for computation.

    Returns:
        List[float]: List of BERTScore F1 values.
    """
    _, _, f1 = score(
        candidate_outputs,
        reference_outputs,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
    )
    return list(f1.numpy())


def compute_cosine_similarity(
    candidate_outputs: List[str],
    reference_outputs: List[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 4,
) -> List[float]:
    """
    Compute cosine similarity between candidate and reference texts using embeddings.

    Parameters:
        candidate_outputs (List[str]): Candidate texts.
        reference_outputs (List[str]): Reference texts.
        embedding_model (SentenceTransformer): Embedding model.
        batch_size (int): Batch size for encoding.

    Returns:
        List[float]: List of cosine similarity values.
    """
    candidate_embeddings = embedding_model.encode(
        candidate_outputs, convert_to_tensor=True, batch_size=batch_size
    )
    reference_embeddings = embedding_model.encode(
        reference_outputs, convert_to_tensor=True, batch_size=batch_size
    )
    similarities = [
        util.pytorch_cos_sim(c, r).item()
        for c, r in zip(candidate_embeddings, reference_embeddings)
    ]
    return similarities


def compute_bleu(
    candidate_outputs: List[str], reference_outputs: List[str]
) -> List[float]:
    """
    Compute BLEU scores for candidate texts against reference texts.

    Parameters:
        candidate_outputs (List[str]): Candidate texts.
        reference_outputs (List[str]): Reference texts.

    Returns:
        List[float]: List of BLEU scores.
    """
    return [
        sentence_bleu([ref.split()], cand.split())
        for cand, ref in zip(candidate_outputs, reference_outputs)
    ]


def compute_rouge(candidate_outputs: List[str], reference_outputs: List[str]) -> float:
    """
    Compute the ROUGE-L score for candidate texts against reference texts.

    Parameters:
        candidate_outputs (List[str]): Candidate texts.
        reference_outputs (List[str]): Reference texts.

    Returns:
        float: ROUGE-L score.
    """
    rouge = load("rouge")
    results = rouge.compute(predictions=candidate_outputs, references=reference_outputs)
    return results["rougeL"]


def compute_perplexity(texts: List[str], tokenizer: Any, lm_model: Any) -> List[float]:
    """
    Compute perplexity for a list of texts using a language model.

    Parameters:
        texts (List[str]): List of texts.
        tokenizer (Any): Tokenizer for the model.
        lm_model (Any): Language model.

    Returns:
        List[float]: List of perplexity scores.
    """
    ppl_scores: List[float] = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = lm_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
        ppl_scores.append(torch.exp(loss).item())
    return ppl_scores


def compute_readability(texts: List[str]) -> List[float]:
    """
    Compute readability (Flesch Reading Ease) for a list of texts.

    Parameters:
        texts (List[str]): List of texts.

    Returns:
        List[float]: List of readability scores.
    """
    scores: List[float] = []
    for text in texts:
        measures = readability.getmeasures(text, lang="en")
        score_value = measures["readability grades"]["FleschReadingEase"]
        scores.append(score_value)
    return scores
