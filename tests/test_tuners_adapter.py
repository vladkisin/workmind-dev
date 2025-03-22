import pytest


class DummyDataset:
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {"input": "dummy input", "label": 0}


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def __call__(
        self, text, return_tensors="pt", padding=True, truncation=True, max_length=None
    ):
        import torch

        return {"input_ids": torch.ones((1, 10), dtype=torch.long)}


def dummy_compute_metrics(eval_prediction):
    return {"accuracy": 1.0}


class DummyModel:
    def add_adapter(self, adapter_name, config):
        pass

    def add_classification_head(self, adapter_name, num_labels, id2label):
        pass

    def train_adapter(self, adapter_name):
        pass

    def add_classification_head(self, adapter_name, num_labels, id2label):
        pass


def test_adapter_finetuner_prepare_model(monkeypatch):
    from workmind.tuners.adapter import AdapterFineTuner

    monkeypatch.setattr(
        "workmind.tuners.adapter.AutoAdapterModel.from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )
    train_dataset = DummyDataset()
    eval_dataset = DummyDataset()
    tokenizer = DummyTokenizer.from_pretrained("dummy-model")
    tuner = AdapterFineTuner(
        model_name_or_path="dummy-model",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        adapter_name="dummy-adapter",
        num_labels=2,
        compute_metrics=dummy_compute_metrics,
    )
    tuner.prepare_model()
    assert tuner.model is not None
