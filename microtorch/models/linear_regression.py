from microtorch.nn.base_model import Model
from microtorch.nn.linear import Linear


class LinearRegression(Model):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self._layers = [self.linear]

    @property
    def layers(self):
        return self._layers

    def forward(self, x):
        y = self.linear.forward(x)
        return y
