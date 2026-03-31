from autograd.autograd import Grad
import numpy as np
from models.mlp import MLP

x = Grad.rand((3, 3))
y = np.random.rand(3, 1)

model = MLP(in_size=3, out_size=1, hidden_layers=[10])

y_pred = model(x)


def loss(y_pred, y):
    return ((y_pred - y) ** 2).mean()

l = loss(y_pred, y)

print(model.parameters())
l.backprop()
print(model.parameters())

