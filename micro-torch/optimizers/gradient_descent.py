from autograd.autograd import Grad
from optimizers import BaseOptimizer

class GradientDescent(BaseOptimizer):
    def __init__(self, params: list[Grad], lr = 0.01):
        super().__init__(params, lr)

    def step(self, zero_grad=True):
        for param in self.params:
            param.val -= self.lr * param.grad
            if zero_grad:
                param.grad = 0