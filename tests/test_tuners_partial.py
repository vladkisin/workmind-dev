import pytest
from workmind.tuners.partial import PartiallyUnfrozenClsFineTuner


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
    def named_parameters(self):
        class DummyParam:
            def __init__(self, name):
                self.name = name
                self.requires_grad = False

        return [
            ("dummy_layer.weight", DummyParam("dummy_layer.weight")),
            ("other.weight", DummyParam("other.weight")),
        ]


def test_partially_unfrozen_finetuner_prepare_model(monkeypatch):
    from transformers import AutoModelForSequenceClassification

    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )
    train_dataset = DummyDataset()
    eval_dataset = DummyDataset()
    tokenizer = DummyTokenizer.from_pretrained("dummy-model")
    tuner = PartiallyUnfrozenClsFineTuner(
        model_name_or_path="dummy-model",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        layers_to_unfreeze=("dummy_layer",),
        compute_metrics=dummy_compute_metrics,
        num_labels=2,
    )
    tuner.prepare_model()
    assert tuner.model is not None
