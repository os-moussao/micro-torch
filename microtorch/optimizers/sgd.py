from microtorch.autograd.autograd import Grad
from microtorch.nn.base_model import Model
from microtorch.optimizers.base import Optimizer


class SGD(Optimizer):
    def __init__(self, model: Model, lr = 0.01):
        super().__init__(model, lr)

    def step(self):
        for param in self.parameters():
            param.val -= self.lr * param.grad