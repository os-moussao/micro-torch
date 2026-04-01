from microtorch.autograd.autograd import Grad
from microtorch.functions.sigmoid import sigmoid
from microtorch.layers.base import Layer


class Sigmoid(Layer):
    def forward(self, x):
        return sigmoid(x)

    def parameters(self) -> list[Grad]:
        return []