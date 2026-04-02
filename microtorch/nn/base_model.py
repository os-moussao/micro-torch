from abc import ABC, abstractmethod
from microtorch.autograd.autograd import Grad
from microtorch.nn.base_layer import Layer


class Model(Layer, ABC):
    def __init__(self):
        super().__init__()

    # todo: use an efficient way to store layers and params
    @property
    @abstractmethod
    def layers(self) -> list[Layer]:
        """Any model must set its layers here."""
        pass

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # todo: enhance perf, use yield
    def parameters(self) -> list[Grad]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
