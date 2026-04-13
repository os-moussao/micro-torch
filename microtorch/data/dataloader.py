import random
from typing import Optional
import numpy as np


class DataLoader():
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        shuffle=False,
        seed: Optional[int] = None
    ):
        assert len(X) == len(y), "Lengths of X and y should be equal"
        self.X = X.copy()
        self.y = y.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            rng = np.random if self.seed is None else np.random.default_rng(
                self.seed)
            rng.shuffle(self.X)
            rng.shuffle(self.y)

        for i in range(0, len(self.X), self.batch_size):
            yield (self.X[i: i + self.batch_size], self.y[i: i + self.batch_size])

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size
