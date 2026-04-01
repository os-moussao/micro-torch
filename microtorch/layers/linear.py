from typing import Literal, Optional
from microtorch.autograd.autograd import Grad
from microtorch.functions.sigmoid import sigmoid
from microtorch.layers.base import Layer

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias=True, activation: Optional[Literal['relu', 'sigmoid']] = None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Grad.rand((in_features, out_features))
        self.bias = Grad.rand((1, out_features)) if bias else None
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
        params: list[Grad] = self.weight.flatten().tolist()
        if self.bias is not None:
            params.extend(self.bias.flatten().tolist())
        return params
