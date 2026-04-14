from microtorch.nn.base_model import Model
from microtorch.optimizers.base import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, model: Model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._mt = [np.float64(0)] * len(self)
        self._vt = [np.float64(0)] * len(self)
        self._t = 0

    def step(self):
        self._t += 1
        for i, (param, _mt, _vt) in enumerate(zip(self.parameters(), self._mt, self._vt)):
            assert param.grad is not None

            g = param.grad

            _mt = self.beta1 * _mt + (1 - self.beta1) * g
            _vt = self.beta2 * _vt + (1 - self.beta2) * (g ** 2)

            self._mt[i] = _mt
            self._vt[i] = _vt

            mt = _mt / (1 - self.beta1 ** self._t)
            vt = _vt / (1 - self.beta2 ** self._t)

            param.val -= self.lr * mt / (np.sqrt(vt) + self.eps)
