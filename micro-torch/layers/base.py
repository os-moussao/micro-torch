from abc import ABC, abstractmethod
from autograd import Grad

class Base(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def parameters(self) -> list[Grad]:
        pass

    def __call__(self, x):
        return self.forward(x)
