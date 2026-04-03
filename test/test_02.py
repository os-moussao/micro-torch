from microtorch.functions import cross_entropy_loss
import numpy as np

y = np.array([1, 0, 0, 0]).reshape(1, -1)
y_pred_1 = np.array([0.9, 0.05, 0.03, 0.02]).reshape(1, -1)
y_pred_2 = np.array([0.1, 0.1, 0.1, 0.7]).reshape(1, -1)

loss_1 = cross_entropy_loss(y_pred_1, y) # low loss
loss_2 = cross_entropy_loss(y_pred_2, y) # high loss
print(f"Cross-Entropy Loss 1: {loss_1}")
print(f"Cross-Entropy Loss 2: {loss_2}")

assert(loss_1 < loss_2), "Cross-Entropy Loss should be lower for better predictions"