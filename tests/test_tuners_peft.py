import pytest
from workmind.tuners.peft import LoraCausalFineTuner


class DummyDataset:
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {"input": "dummy input", "label": 0}


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.pad_token_id = 0
        self.eos_token_id = 0

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def __call__(
        self, text, return_tensors="pt", padding=True, truncation=True, max_length=None
    ):
        import torch

        return {"input_ids": torch.ones((1, 10), dtype=torch.long)}


def test_lora_causal_finetuner_prepare_model(monkeypatch):
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: DummyModel()
    )
    tokenizer = DummyTokenizer.from_pretrained("dummy-model")
    train_dataset = DummyDataset()
    eval_dataset = DummyDataset()
    tuner = LoraCausalFineTuner(
        model_name_or_path="dummy-model",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        lora_target_modules=["dummy_module"],
    )
    tuner.prepare_model()
    assert tuner.model is not None


class DummyModel:
    def __init__(self):
        self.config = type("Config", (), {})()
        self.config.use_cache = False
        self.config.pretraining_tp = 1
