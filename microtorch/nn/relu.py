from microtorch.autograd.autograd import Grad
from microtorch.nn.base_layer import Layer


class ReLU(Layer):
    def forward(self, x):
        return Grad.relu(x)

    def parameters(self) -> list[Grad]:
        return []
