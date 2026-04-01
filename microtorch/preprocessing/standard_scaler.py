import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, X: np.ndarray):
        assert len(X.shape) == 2, "Input must be a 2D array"
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
