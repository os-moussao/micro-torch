import numpy as np
from microtorch.autograd.autograd import Grad

def get_exp(y: np.ndarray) -> np.ndarray:
    return Grad.exp(y) if y.dtype == 'object' else np.exp(y)

def softmax(y: np.ndarray) -> np.ndarray:
    y_max = y.max(axis=1, keepdims=True)
    exp = get_exp(y - y_max)
    return exp / exp.sum(axis=1, keepdims=True)