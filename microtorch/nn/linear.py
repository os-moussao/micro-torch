from typing import Literal, Optional
from microtorch.autograd.autograd import Grad
from microtorch.functions.sigmoid import sigmoid
from microtorch.nn.base_layer import Layer
import numpy as np


class Linear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        activation: Optional[Literal['relu', 'sigmoid']] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self._initialize_weights(activation)
        self.bias = Grad.zeros((1, out_features)) if bias else None
        self.activation = activation

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y += self.bias

        if self.activation == 'relu':
            y = Grad.relu(y)
        elif self.activation == 'sigmoid':
            y = sigmoid(y)

        return y

    def parameters(self):
        params = self.weight.flatten().tolist()
        if self.bias is not None:
            params.extend(self.bias.flatten().tolist())
        return params

    def _initialize_weights(self, activation):
        # use he initialization for relu activation
        if activation == 'relu':
            raw_weights = np.random.normal(0, np.sqrt(
                2 / self.in_features), (self.in_features, self.out_features))

        # use xavier initialization for sigmoid/tanh activations
        else:
            raw_weights = np.random.normal(0, np.sqrt(
                1 / self.in_features), (self.in_features, self.out_features))

        return Grad.array(raw_weights)
