import numpy as np
from microtorch.autograd.autograd import Grad


def softmax(y: np.ndarray) -> np.ndarray:
    y_exp = Grad.exp(y) if y.dtype == 'object' else np.exp(y)
    return y_exp / y_exp.sum()
