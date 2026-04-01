from microtorch.autograd.autograd import Grad
from microtorch.layers.base import Layer


class ReLU(Layer):
    def forward(self, x):
        return Grad.relu(x)

    def parameters(self) -> list[Grad]:
        return []
