from abc import ABC, abstractmethod
from typing import Dict


class AbstractFineTuner(ABC):
    """
    Defines the interface for different fine-tuning approaches.
    """

    @abstractmethod
    def prepare_model(self) -> None:
        """
        Prepare the model (load from checkpoint, add adapters/LoRA layers, freeze/unfreeze layers, etc.).
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Execute the training process.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on a validation/test set and return metrics.
        """
        pass
