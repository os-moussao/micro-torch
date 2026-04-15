from abc import ABC, abstractmethod
from microtorch.autograd.autograd import Grad
from microtorch.nn.base_layer import Layer
import pickle


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

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model: Model = pickle.load(file)
            return model
