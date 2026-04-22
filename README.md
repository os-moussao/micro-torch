# Micro Torch
Creating a micro version of [PyTorch](https://pytorch.org/) from scratch ([numpy only](./requirements/microtorch-requirements.txt)).

It contains the necessary building blocks for making ML models, such as autograd, optimizers, and NN layers in a pytorch-like style.

The goal is to understand how ML models and Neural Networks work under the hood and have fun building things from scratch.

# How to use Micro Torch

Here are some examples demonstrating how to use the different components of the `microtorch` API.

## 1. The Autograd Engine

You can use the `Grad` class to perform scalar operations and automatically compute gradients via backpropagation.

```python
from microtorch.autograd import Grad

# Create scalar values requiring gradients
x = Grad(2.0, required_grad=True)
y = Grad(3.0, required_grad=True)

# Perform mathematical operations
z = (x * y) + (x ** 2)

# Compute gradients
z.backprop()

print(f"z = {z.val}")      # Output: 10.0
print(f"dz/dx = {x.grad}") # Output: 7.0
print(f"dz/dy = {y.grad}") # Output: 2.0
```

## 2. Creating a Neural Network (using the `MLP` Model)

The built-in `MLP` model in the `models` module lets you quickly instantiate a Multi-Layer Perceptron.

(Other builtin models: `LinearRegression`, `LogisticRegression`)

```python
from microtorch import models
from microtorch.functions import mean_squared_error
from microtorch.optimizers import SGD
import numpy as np

np.random.seed(1337)

# Sample data
X = np.random.randn(10, 3)
Y = 2 * X.sum(axis=1, keepdims=True) + 1

# Define NN model using the MLP utility
model = models.MLP(
    in_size=3,
    out_size=1,
    hidden_layers=[10],
    activations=['relu']
)

# Setup optimizer (SGD or Adam)
optimizer = SGD(model, lr=0.01)

# Training step (Forward pass)
y_pred = model(X)
loss = mean_squared_error(y_pred, Y)
print(f"Loss Before Optimization: {loss.val}")

# Backpropagation & optimization
loss.backprop()
optimizer.step()
optimizer.zero_grad()

# Forward pass after optimization
y_pred = model(X)
loss = mean_squared_error(y_pred, Y)
print(f"Loss After Optimization: {loss.val}")
```

## 3. Creating a Custom Neural Network

For more flexibility, you can create a custom neural network without using the `MLP` utility. By inheriting `Model` class, you can layout the architecture and define the `forward` pass logic yourself.

```python
from microtorch.models.base_model import Model
from microtorch import nn

# Define custom model
class CustomNN(Model):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, 10)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(10, out_features)

    # Always expose your layers via the `layers` property
    # so the optimizer can find their parameters
    @property
    def layers(self):
        return [self.lin1, self.relu, self.lin2]
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x
```

## Other features

More features are available, such as additional activation functions, optimizers (Adam), loss functions as well as utils for data handling: check the [testing directory](./test) for complete examples and usage of the API.