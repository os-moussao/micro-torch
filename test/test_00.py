from microtorch import nn
from microtorch.functions import mean_squared_error
from microtorch.optimizers import GradientDescent
import numpy as np

np.random.seed(1337)

x = np.random.rand(4, 3)
y = x.sum(axis=1).reshape(-1, 1)

model = nn.MLP(in_size=3, out_size=1, hidden_layers=[10])
loss = mean_squared_error
optimizer = GradientDescent(model.parameters(), lr=0.01)

lr = 0.01
ep = 30

for i in range(ep):
    y_pred = model(x)
    l = loss(y_pred, y)

    # if loss is getting smaller, then the model works
    print(f"Loss: {l}")

    # Backpropagation
    l.backprop()

    optimizer.step()
    optimizer.zero_grad()
