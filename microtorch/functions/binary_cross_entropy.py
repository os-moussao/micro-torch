from microtorch.autograd.autograd import Grad
import numpy as np

def BCELoss(y_pred: np.ndarray, y_true: np.ndarray):
    eps = 1e-12
    if y_pred.dtype == 'object':
        y_pred = Grad.clip(y_pred, eps, 1 - eps)
        neg_log = - (y_true * Grad.log(y_pred) + (1 - y_true) * Grad.log(1 - y_pred)).sum(axis=1)
        return neg_log.mean()
    else:
        y_pred = np.clip(y_pred, eps, 1 - eps)
        neg_log = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).sum(axis=1)
        return neg_log.mean()
