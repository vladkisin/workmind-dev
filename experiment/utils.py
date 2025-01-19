import pandas as pd
from sklearn.metrics import classification_report


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
    # Create a DataFrame for easy manipulation
    data = pd.DataFrame({
        'user_id': user_ids,
        'predicted': predicted_sentiments,
        'real': real_sentiments
    })

    # Aggregate sentiments by user
    user_level = data.groupby('user_id').agg({
        'predicted': lambda x: 'negative' if 'negative' in x.values else 'positive',
        'real': lambda x: 'negative' if 'negative' in x.values else 'positive'
    }).reset_index()

    # Calculate classification metrics
    report = classification_report(user_level['real'], user_level['predicted'], output_dict=True)
    return report
