from microtorch.autograd.autograd import Grad
import numpy as np

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray):
    eps = 1e-10
    if y_pred.dtype == 'object':
        y_pred = Grad.clip(y_pred, eps, 1)
        neg_log = - (y_true * Grad.log(y_pred)).sum(axis=1)
        return neg_log.mean()
    else:
        y_pred = np.clip(y_pred, eps, 1)
        neg_log = - (y_true * np.log(y_pred)).sum(axis=1)
        return neg_log.mean()

