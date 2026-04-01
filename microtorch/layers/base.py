from abc import ABC, abstractmethod
from microtorch.autograd.autograd import Grad


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def parameters(self) -> list[Grad]:
        pass

    def __call__(self, x):
        return self.forward(x)
