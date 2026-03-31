from autograd.autograd import Grad
import numpy as np
from optimizers import GradientDescent
from functions import mse
import nn

np.random.seed(1337)

x = np.random.rand(4, 3)
y = x.sum(axis=1).reshape(-1, 1)

model = nn.MLP(in_size=3, out_size=1, hidden_layers=[10])
loss = mse
optimizer = GradientDescent(model.parameters(), lr=0.01)

lr = 0.01
ep = 30

for i in range(ep):
    y_pred = model(x)
    l = loss(y_pred, y)
    print(f"Loss: {l}")

    # Backpropagation
    l.backprop()

    optimizer.step()
    optimizer.zero_grad()
