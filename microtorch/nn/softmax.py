from microtorch.autograd.autograd import Grad
from microtorch.functions.softmax import softmax
from microtorch.nn.base_layer import Layer


class Softmax(Layer):
    def forward(self, x):
        return softmax(x)

    def parameters(self) -> list[Grad]:
        return []
