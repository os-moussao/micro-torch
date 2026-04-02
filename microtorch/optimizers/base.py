from abc import ABC, abstractmethod
from typing import List
from microtorch.autograd.autograd import Grad


class Optimizer(ABC):
    def __init__(self, params: List[Grad], lr: float = 0.01):
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self):
        """Update parameters based on their gradients."""
        pass

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.params:
            param.grad = 0
