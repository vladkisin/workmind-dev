import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_and_split_gd(df: pd.DataFrame):
    df = df[df['current'].str.contains("Current Employee")]
    df.fillna('', inplace=True)
    df['review'] = df.apply(lambda row: row['headline'] + '\n\nPros:\n' + row['pros'] + '\n\nCons:\n' + row['cons'],
                            axis=1)

    filtered_df = df.loc[
        ((df['recommend'] == 'v') & (df['overall_rating'].between(4, 5))) |
        ((df['recommend'] == 'x') & (df['overall_rating'].between(1, 2))) |
        ((df['recommend'] == 'o') & (df['overall_rating'].between(3, 3)))
        ]
    filtered_df['text_length'] = filtered_df['review'].str.len()
    lower_bound = filtered_df['text_length'].quantile(0.05)
    upper_bound = filtered_df['text_length'].quantile(0.95)
    filtered_df = filtered_df[
        (filtered_df['text_length'] >= lower_bound) & (filtered_df['text_length'] <= upper_bound)
        ]
    filtered_df['label'] = filtered_df['recommend'].map({'v': 2, 'x': 0, 'o': 1}).tolist()
    filtered_df['text'] = filtered_df['review']
    train_df, temp_df = train_test_split(filtered_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=1 / 3, random_state=42)

    return train_df, val_df, test_df


def chunk_text_by_sentences(sentences, min_chunk=500, max_chunk=1900, target_overlap=160):
    chunks = []
    current_chunk = []
    current_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)

        if sentence_length >= max_chunk:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(sentence)
            i += 1
            continue

        if current_length + sentence_length + (1 if current_chunk else 0) > max_chunk:
            if current_length < min_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length + (1 if current_chunk[:-1] else 0)
                i += 1
                continue
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                overlap_chunk = []
                overlap_length = 0
                j = len(current_chunk) - 1
                while j >= 0 and (overlap_length < target_overlap or not overlap_chunk):
                    overlap_chunk.insert(0, current_chunk[j])
                    overlap_length += len(current_chunk[j]) + 1
                    j -= 1

                current_chunk = overlap_chunk.copy()
                current_length = sum(len(s) for s in current_chunk) + (len(current_chunk)-1)
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
