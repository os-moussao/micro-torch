"""
Testing linear regression model
"""
from microtorch import models
from microtorch.functions import mean_squared_error
from microtorch.optimizers import SGD
import numpy as np

np.random.seed(1337)

x = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2]
])
y = x.sum(axis=1).reshape(-1, 1)


model = models.LinearRegression(in_features=3, out_features=1)
loss = mean_squared_error
optimizer = SGD(model.parameters(), lr=0.01)

ep = 1000

for i in range(ep):
    y_pred = model(x)
    l = loss(y_pred, y)

    # if loss is converging to 0, then the model works
    print(f"Loss: {l}")

    # backpropagation & optimization
    l.backprop()
    optimizer.step()
    optimizer.zero_grad()

# should be close to 3
x_test = np.ones((1, 3))
y_test = model(x_test)
print(f"Prediction for {x_test}: {y_test}")
