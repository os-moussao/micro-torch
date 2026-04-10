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

        self._parameters: list[Grad] = self.weight.flatten().tolist()
        if self.bias is not None:
            self._parameters.extend(self.bias.flatten().tolist())

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
        return self._parameters

    def _initialize_weights(self, activation):
        # use he initialization for relu activation
        if activation == 'relu':
            bound = np.sqrt(6.0 / self.in_features)

        # use xavier initialization for sigmoid/tanh activations
        elif activation == 'sigmoid':
            bound = np.sqrt(6.0 / (self.in_features + self.out_features))

        else:
            return Grad.rand((self.in_features, self.out_features))

        raw_weights = np.random.uniform(-bound, bound,
                                        size=(self.in_features, self.out_features))

        return Grad.array(raw_weights)
