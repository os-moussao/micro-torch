from microtorch.models.base_model import Model
from microtorch.nn.linear import Linear
from microtorch.nn.sigmoid import Sigmoid


class LogisticRegression(Model):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.lin = Linear(input_dim, 1)
        self.sigmoid = Sigmoid()
        self._layers = [self.lin, self.sigmoid]

    @property
    def layers(self):
        return self._layers

    def forward(self, x):
        x = self.lin(x)
        x = self.sigmoid(x)
        return x
