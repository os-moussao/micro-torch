from autograd import Grad


def sigmoid(x):
    return 1 / (1 + Grad.exp(-x))
