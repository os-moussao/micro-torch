from abc import ABC, abstractmethod
from typing import List
from microtorch.autograd.autograd import Grad
from microtorch.nn.base_model import Model


class Optimizer(ABC):
    def __init__(self, model: Model, lr: float = 0.01):
        self.lr = lr
        self.model = model
        self.len = len(model.parameters())

    @abstractmethod
    def step(self):
        """Update parameters based on their gradients."""
        pass

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.parameters():
            param.grad = 0

    def __iter__(self):
        for param in self.parameters():
            yield param

    def __len__(self):
        return self.len
