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
