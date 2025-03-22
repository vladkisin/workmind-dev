# WorkMind: HR Sentiment Analysis and Intervention Generation Toolkit

**WorkMind** is a Python toolkit designed to analyze employee sentiment and generate structured HR interventions. It utilizes advanced NLP models from Hugging Face, supporting both standard generation methods and Retrieval-Augmented Generation (RAG).

---

## 📌 Overview

The toolkit provides two main components:

- **Sentiment Analysis:**
  - Classification-based sentiment analysis
  - Natural Language Inference (NLI)-based sentiment analysis
  - Large Language Model (LLM)-based sentiment analysis

- **Intervention Generation:**
  - Direct intervention generation via causal language models
  - RAG-based interventions using custom knowledge bases (with LlamaIndex)

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vladkisin/workmind-dev.git
cd workmind-dev
```

### 2. Set up Python Environment

It is recommended to use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install WorkMind as a Local Package

```bash
pip install -e .
```

---

## 💡 Basic Usage

### Sentiment Analysis

WorkMind offers sentiment analyzers using a convenient factory method:

```python
from workmind.analyzers import get_analyzer

analyzer = get_analyzer(
    inference_type="classification",
    model_name="textattack/bert-base-uncased-SST-2"
)

texts = [
    "I am very happy with my job!",
    "I'm exhausted and frustrated."
]

predictions = analyzer.predict(texts)

for prediction in predictions:
    print(f"Text: {prediction['text']}")
    print(f"Sentiment: {prediction['predicted_sentiment']}\n")
```

### Supported Zero-Shot Sentiment Models:

| Model Name                                                     | Type            | Sentiment Labels               |
|----------------------------------------------------------------|-----------------|--------------------------------|
| `textattack/bert-base-uncased-SST-2`                           | Classification  | Negative, Positive             |
| `dipawidia/xlnet-base-cased-product-review-sentiment-analysis` | Classification  | Negative, Positive             |
| `siebert/sentiment-roberta-large-english`                      | Classification  | Negative, Positive             |
| `nlptown/bert-base-multilingual-uncased-sentiment`             | Classification  | Negative, Neutral, Positive    |
| `roberta-large-mnli`                                           | NLI             | Negative, Neutral, Positive    |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`     | NLI             | Negative, Neutral, Positive    |

---

### Generating HR Interventions

Generate structured HR interventions from employee messages:

```python
from workmind.generators.interventions import InterventionGenerator

generator = InterventionGenerator(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    batch_size=2
)

email_batches = [
    ["My workload is too heavy, and I feel unsupported."],
    ["I enjoy my tasks, and my team is very helpful."]
]

interventions = generator.predict(email_batches)

for intervention in interventions:
    print(intervention)
```

### Supported Intervention Generation Models:

- `microsoft/Phi-3-mini-4k-instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `tiiuae/Falcon3-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- Any compatible causal language model on Hugging Face supporting chat/instruct prompts (system and user context).

---

## 🔍 Retrieval-Augmented Generation (RAG)

For advanced context-aware interventions, WorkMind supports RAG leveraging your own knowledge base.

Check the [provided notebook](jnbs/evalragapp.ipynb) for a practical example.

It covers:
- Creating/loading RAG indices
- Defining prompts/templates
- Generating context-enhanced HR interventions

---

## 🛠️ Fine-tuning Capabilities

WorkMind supports fine-tuning sentiment and generation models using advanced methods:

### Supported Tuning Strategies:

- **LoRA (Low-Rank Adaptation)** for causal LLMs
- **Adapters** for classification models
- **Partial layer unfreezing** for transformers-based classifiers

#### Example: LoRA Fine-Tuning for Causal LM

```python
from workmind.tuners.peft import LoraCausalFineTuner

tuner = LoraCausalFineTuner(
    model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
    train_dataset=train_data,
    eval_dataset=eval_data,
    num_train_epochs=3,
    learning_rate=1e-4,
)

tuner.prepare_model()
tuner.train()
metrics = tuner.evaluate()
print(metrics)
```

#### Example: Adapter-based Fine-Tuning for Classification

```python
from workmind.tuners.adapter import AdapterFineTuner

tuner = AdapterFineTuner(
    model_name_or_path="textattack/bert-base-uncased-SST-2",
    train_dataset=train_data,
    eval_dataset=eval_data,
    adapter_name="sentiment-adapter",
    num_labels=2,
    num_train_epochs=5,
)

tuner.prepare_model()
tuner.train()
metrics = tuner.evaluate()
print(metrics)
```

---

## 📊 Experiments and Logging

WorkMind seamlessly integrates with:

- **MLflow**
- **Weights & Biases (WandB)**

### WandB Sentiment Analysis Experiment Example

```python
from workmind.analyzers import get_analyzer
from workmind.experiment.wandb.sentiment import SentimentExperiment

analyzer = get_analyzer(
    inference_type="classification",
    model_name="textattack/bert-base-uncased-SST-2"
)

texts = ["I feel undervalued.", "I love working here."]
labels = ["negative", "positive"]

with SentimentExperiment(
    analyzer, 
    experiment_name="BERT-SST2-sentiment",
    true_labels=labels,
    log_predictions=True,
    project_name="workmind-sentiment",
) as exp:
    exp.evaluate(texts)
```

- Results are automatically logged and visualized in your WandB dashboard.

---

## 📁 Project Structure

```
workmind/
├── analyzers/
│   └── sentiment/
├── generators/
├── data_processing/
├── experiment/
│   ├── mlflow/
│   └── wandb/
├── tuners/

```

---

## ✅ Contributing & Issues

- Contributions are welcome — submit Pull Requests!
- Issues/feature requests: [GitHub Issues](https://github.com/vladkisin/workmind-dev/issues)

---

## ❓ Need Help?

If anything's unclear or missing, feel free to open an issue or contact directly!

---

**Happy HR analytics! 🚀**