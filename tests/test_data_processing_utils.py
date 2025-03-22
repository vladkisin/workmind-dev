import pandas as pd
import pytest
from workmind.data_processing.utils import (
    preprocess_and_split_gd,
    chunk_text_by_sentences,
)


def test_preprocess_and_split_gd():
    data = {
        "current": ["Current Employee"] * 6 * 2,
        "headline": [
            "Great place",
            "Needs improvement",
            "Excellent",
            "Very good",
            "Superb",
            "Not used",
        ]
        * 2,
        "pros": [
            "Good benefits",
            "Friendly",
            "Supportive",
            "Efficient",
            "Caring",
            "N/A",
        ]
        * 2,
        "cons": [
            "Long hours",
            "Poor management",
            "Limited growth",
            "High workload",
            "None",
            "N/A",
        ]
        * 2,
        "recommend": ["v", "x", "v", "v", "x", "o"] * 2,
        "overall_rating": [5, 1, 5, 5, 1, 3] * 2,
    }
    df = pd.DataFrame(data)
    train_df, val_df, test_df = preprocess_and_split_gd(df)
    for d in [train_df, val_df, test_df]:
        assert all(d["current"] == "Current Employee")
    for review in train_df["review"]:
        assert "Pros:" in review and "Cons:" in review


def test_chunk_text_by_sentences():
    sentences = [
        f"This is sentence number {i}, with some additional text to reach a reasonable length."
        for i in range(1, 40)
    ]
    chunks = chunk_text_by_sentences(
        sentences, min_chunk=50, max_chunk=100, target_overlap=20
    )

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    joined_text = " ".join(chunks)
    for sentence in sentences:
        assert sentence in joined_text, f"Missing sentence: {sentence}"

    for chunk in chunks:
        assert len(chunk) <= 110, f"Chunk length {len(chunk)} exceeds expected maximum"

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1}: {chunk}")
