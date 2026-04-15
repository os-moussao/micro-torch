from typing import Literal, Optional
from microtorch.nn.base_layer import Layer
from microtorch.nn.linear import Linear
from microtorch.models.base_model import Model


class MLP(Model):
    """Multi-Layer Perceptron (MLP) model."""

    def __init__(
        self, in_size: int,
        out_size: int,
        hidden_layers: list[int],
        activations: Optional[list[Literal['sigmoid', 'relu']]] = None
    ):
        assert len(hidden_layers) > 0, "At least one hidden layer is required."
        assert activations is None or len(hidden_layers) == len(
            activations), "Activations should match hidden layers length"
        super().__init__()

        self._layers: list[Layer] = []

        for i in range(len(hidden_layers)):
            self._layers.append(Linear(
                in_features=hidden_layers[i-1] if i != 0 else in_size,
                out_features=hidden_layers[i],
                activation=activations[i] if activations is not None else 'relu')
            )

        self._layers.append(Linear(
            in_features=hidden_layers[-1],
            out_features=out_size,
            activation=None)
        )

    @property
    def layers(self):
        return self._layers

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
