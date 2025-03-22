import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

LOWER_QUANTILE: float = 0.05
UPPER_QUANTILE: float = 0.95


def preprocess_and_split_gd(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess Glassdoor review data and split into train, validation, and test sets.

    Parameters:
        df (pd.DataFrame): DataFrame containing review data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames.
    """
    df = df[df["current"].str.contains("Current Employee")]
    df.fillna("", inplace=True)
    df["review"] = df.apply(
        lambda row: row["headline"]
        + "\n\nPros:\n"
        + row["pros"]
        + "\n\nCons:\n"
        + row["cons"],
        axis=1,
    )
    filtered_df = df.loc[
        ((df["recommend"] == "v") & (df["overall_rating"].between(4, 5)))
        | ((df["recommend"] == "x") & (df["overall_rating"].between(1, 2)))
        | ((df["recommend"] == "o") & (df["overall_rating"].between(3, 3)))
    ]
    filtered_df["text_length"] = filtered_df["review"].str.len()
    lower_bound = filtered_df["text_length"].quantile(LOWER_QUANTILE)
    upper_bound = filtered_df["text_length"].quantile(UPPER_QUANTILE)
    filtered_df = filtered_df[
        (filtered_df["text_length"] >= lower_bound)
        & (filtered_df["text_length"] <= upper_bound)
    ]
    filtered_df["label"] = (
        filtered_df["recommend"].map({"v": 2, "x": 0, "o": 1}).tolist()
    )
    filtered_df["text"] = filtered_df["review"]
    train_df, temp_df = train_test_split(filtered_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=1 / 3, random_state=42)
    return train_df, val_df, test_df


def chunk_text_by_sentences(
    sentences: list,
    min_chunk: int = 500,
    max_chunk: int = 1900,
    target_overlap: int = 160,
) -> list:
    """
    Split a list of sentences into chunks based on minimum, maximum lengths and target overlap.

    Parameters:
        sentences (list): List of sentence strings.
        min_chunk (int): Minimum allowed chunk length.
        max_chunk (int): Maximum allowed chunk length.
        target_overlap (int): Target number of overlapping characters between chunks.

    Returns:
        list: List of text chunks.
    """
    chunks: list = []
    current_chunk: list = []
    current_length: int = 0
    i: int = 0
    while i < len(sentences):
        sentence: str = sentences[i]
        sentence_length: int = len(sentence)
        if sentence_length >= max_chunk:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(sentence)
            i += 1
            continue
        additional_space: int = 1 if current_chunk else 0
        if current_length + additional_space + sentence_length > max_chunk:
            if current_length < min_chunk:
                current_chunk.append(sentence)
                current_length += additional_space + sentence_length
                i += 1
                continue
            else:
                chunks.append(" ".join(current_chunk))
                overlap_chunk: list = []
                overlap_length: int = 0
                j: int = len(current_chunk) - 1
                while j >= 0 and (overlap_length < target_overlap or not overlap_chunk):
                    overlap_chunk.insert(0, current_chunk[j])
                    overlap_length += len(current_chunk[j]) + 1
                    j -= 1
                new_overlap_length: int = sum(len(s) for s in overlap_chunk) + (
                    len(overlap_chunk) - 1 if overlap_chunk else 0
                )
                if not overlap_chunk or new_overlap_length >= current_length:
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = overlap_chunk.copy()
                    current_length = new_overlap_length
                continue
        else:
            if current_chunk:
                current_length += 1
            current_chunk.append(sentence)
            current_length += sentence_length
            i += 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
