from abc import ABC, abstractmethod
from typing import Dict


class AbstractFineTuner(ABC):
    """
    Abstract interface for a model fine-tuner.
    """

    @abstractmethod
    def prepare_model(self) -> None:
        """
        Prepare the model (e.g., load checkpoint, add adapters, freeze/unfreeze layers).
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model and return evaluation metrics.
        """
        pass
