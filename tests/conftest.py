import pytest
import torch


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.config = type("DummyConfig", (), {})()
        self.config.use_cache = False
        self.config.num_labels = kwargs.get("num_labels", 2)
        self.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def to(self, device):
        return self

    def __call__(self, **kwargs):

        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        num_labels = self.config.num_labels
        dummy_logits = torch.randn(batch_size, num_labels)

        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits

        return DummyOutput(dummy_logits)

    def generate(self, **kwargs):

        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        dummy_token = torch.zeros((batch_size, 1), dtype=torch.long)
        return torch.cat([input_ids, dummy_token], dim=1)

    def load_adapter(self, adapter_name):
        return "dummy_adapter"

    def set_active_adapters(self, adapter):
        pass

    def train_adapter(self, adapter_name):
        pass

    def add_adapter(self, adapter_name, config):
        pass

    def add_classification_head(self, adapter_name, num_labels, id2label):
        pass


class DummyBatchEncoding(dict):
    def __init__(self, data):
        super().__init__(data)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'DummyBatchEncoding' object has no attribute '{name}'"
            )

    def to(self, device):
        new_data = {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()
        }
        return DummyBatchEncoding(new_data)


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(
        self,
        text,
        text_pair=None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=None,
    ):
        max_length = max_length if max_length is not None else 10
        batch_size = len(text) if isinstance(text, list) else 1
        return DummyBatchEncoding(
            {"input_ids": torch.ones((batch_size, max_length), dtype=torch.long)}
        )

    def batch_decode(self, input_ids, skip_special_tokens=True):
        return ["dummy decoded text" for _ in input_ids]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Return a dummy prompt string.
        return "dummy prompt"

    def decode(self, token_ids, skip_special_tokens=True):
        return "dummy decoded text"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture(autouse=True)
def monkeypatch_transformers(monkeypatch):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoConfig,
    )

    try:
        from adapters import AutoAdapterModel
    except ImportError:
        AutoAdapterModel = None

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer()
    )
    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: type(
            "DummyConfig",
            (),
            {"num_labels": kwargs.get("num_labels", 2), "use_cache": False},
        )(),
    )
    if AutoAdapterModel is not None:
        monkeypatch.setattr(
            AutoAdapterModel,
            "from_pretrained",
            lambda *args, **kwargs: DummyModel(**kwargs),
        )
