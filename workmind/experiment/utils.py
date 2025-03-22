import re
import pandas as pd
import readability
import torch
from sklearn.metrics import classification_report
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
from sentence_transformers import SentenceTransformer, util


def calculate_user_level_metrics(user_ids, predicted_sentiments, real_sentiments):
    """
    Aggregates email sentiments by user and calculates classification metrics on the user level.

    Parameters:
        user_ids (list): List of user IDs corresponding to each email.
        predicted_sentiments (list): List of predicted sentiments for each email ('positive', 'negative', etc.).
        real_sentiments (list): List of real sentiments for each email ('positive', 'negative', etc.).

    Returns:
        dict: Classification report for user-level predictions.
    """
    data = pd.DataFrame(
        {
            "user_id": user_ids,
            "predicted": predicted_sentiments,
            "real": real_sentiments,
        }
    )

    # Aggregate sentiments by user
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

    # Calculate classification metrics
    report = classification_report(
        user_level["real"], user_level["predicted"], output_dict=True
    )
    return report


def clean_text(text):
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def compute_bertscore(
    candidate_outputs, reference_outputs, model_type="roberta-base", batch_size=4
):
    _, _, f1 = score(
        candidate_outputs,
        reference_outputs,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
    )
    return list(f1.numpy())


def compute_cosine_similarity(
    candidate_outputs, reference_outputs, embedding_model, batch_size=4
):
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


def compute_bleu(candidate_outputs, reference_outputs):
    return [
        sentence_bleu([ref.split()], cand.split())
        for cand, ref in zip(candidate_outputs, reference_outputs)
    ]


def compute_rouge(candidate_outputs, reference_outputs):
    rouge = load("rouge")
    results = rouge.compute(predictions=candidate_outputs, references=reference_outputs)
    return results["rougeL"]


def compute_perplexity(texts, tokenizer, lm_model):
    ppl_scores = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = lm_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
        ppl_scores.append(torch.exp(loss).item())
    return ppl_scores


def compute_readability(texts):
    scores = []
    for text in texts:
        measures = readability.getmeasures(text, lang="en")
        score_value = measures["readability grades"]["FleschReadingEase"]
        scores.append(score_value)
    return scores
