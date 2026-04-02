from abc import ABC, abstractmethod
from typing import Generator
from microtorch.autograd.autograd import Grad
from microtorch.nn.base_layer import Layer


class Model(Layer, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def layers(self) -> list[Layer]:
        pass

    def parameters(self) -> list[Grad]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
