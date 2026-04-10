import numpy as np
from microtorch.autograd.autograd import Grad


def softmax(y: np.ndarray) -> np.ndarray:
    y_max = y.max(axis=1, keepdims=True)
    y_exp = Grad.exp(y - y_max) if y.dtype == 'object' else np.exp(y - y_max)
    return y_exp / y_exp.sum(axis=1, keepdims=True)