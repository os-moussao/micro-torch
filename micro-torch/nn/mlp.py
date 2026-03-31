from layers import Linear
from autograd import Grad
from .base import Model

class MLP(Model):
    """Multi-Layer Perceptron (MLP) model."""
    def __init__(self, in_size: int, out_size: int, hidden_layers: list[int] = [32]):
        assert len(hidden_layers) > 0, "At least one hidden layer is required."
        super().__init__()

        for i in range(len(hidden_layers)):
            self.layers.append(Linear(
                in_features=hidden_layers[i-1] if i != 0 else in_size,
                out_features=hidden_layers[i],
                activation='relu')
            )

        self.layers.append(Linear(
            in_features=hidden_layers[-1],
            out_features=out_size,
            activation='relu')
        )
