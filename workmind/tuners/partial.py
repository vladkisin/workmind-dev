from workmind.tuners.basic import BaseClsFineTuner


class PartiallyUnfrozenClsFineTuner(BaseClsFineTuner):
    """
    Fine-tuner that unfreezes only selected layers of a classification model.
    """

    def __init__(
        self,
        layers_to_unfreeze: tuple = ("layer.21", "layer.22", "layer.23", "classifier"),
        **kwargs,
    ) -> None:
        """
        Initialize the partially unfrozen fine-tuner.

        Parameters:
            layers_to_unfreeze (tuple): Tuple of layer name substrings to unfreeze.
            Other parameters are passed to the BaseClsFineTuner.
        """
        super().__init__(**kwargs)
        self.layers_to_unfreeze = layers_to_unfreeze

    def prepare_model(self) -> None:
        """
        Load the model and freeze all layers except those specified.
        """
        super().prepare_model()
        for name, param in self.model.named_parameters():
            if any(layer_id in name for layer_id in self.layers_to_unfreeze):
                param.requires_grad = True
            else:
                param.requires_grad = False
