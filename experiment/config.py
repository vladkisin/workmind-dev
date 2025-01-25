MODELS_CONFIG = {
    "textattack/bert-base-uncased-SST-2": {
        "inference_type": "classification",
        "class_labels": ["negative", "positive"],
        "batch_size": 32,
        "hypothesis_template": None,
        "experiment_name": "BERT CLS Pretrained"
    },
    "dipawidia/xlnet-base-cased-product-review-sentiment-analysis": {
        "inference_type": "classification",
        "class_labels": ["negative", "positive"],
        "batch_size": 32,
        "hypothesis_template": None,
        "experiment_name": "XLNet CLS Pretrained"
    },
    "siebert/sentiment-roberta-large-english": {
        "inference_type": "classification",
        "class_labels": ["negative", "positive"],
        "batch_size": 32,
        "hypothesis_template": None,
        "experiment_name": "Roberta Large CLS Pretrained"
    },
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        "inference_type": "classification",
        "class_labels": ["negative", "negative", "neutral", "positive", "positive"],
        "batch_size": 32,
        "hypothesis_template": None,
        "experiment_name": "BERT Sentiment Pretrained 5 classes"
    },
    "roberta-large-mnli": {
        "inference_type": "nli",
        "class_labels": ["negative", "neutral", "positive"],
        "batch_size": 16,
        "hypothesis_template": "This text conveys a {} sentiment.",
        "experiment_name": "Roberta Large NLI"
    },
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": {
        "inference_type": "nli",
        "class_labels": ["negative", "neutral", "positive"],
        "batch_size": 16,
        "hypothesis_template": "This text conveys a {} sentiment.",
        "experiment_name": "Deberta-v3 Large NLI"
    },
}

