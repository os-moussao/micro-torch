from abc import ABC, abstractmethod
from typing import Generator, Optional
from microtorch.autograd.autograd import Grad
from microtorch.nn.base_layer import Layer


class Model(Layer, ABC):
    def __init__(self):
        super().__init__()

        self._parameters: Optional[list[Grad]] = None

    @property
    @abstractmethod
    def layers(self) -> list[Layer]:
        pass

    def parameters(self) -> list[Grad]:
        if self._parameters is None:
            params = []
            for layer in self.layers:
                params.extend(layer.parameters())
            self._parameters = params

        return self._parameters
