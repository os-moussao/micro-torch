from microtorch.autograd.autograd import Grad
from microtorch.optimizers.base import Optimizer


class SGD(Optimizer):
    def __init__(self, params: list[Grad], lr = 0.01):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.val -= self.lr * param.grad