from microtorch.autograd.autograd import Grad
from microtorch.layers.base import Layer


class Model(Layer):
    def __init__(self):
        super().__init__()
        self.layers: list[Layer] = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Grad]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
