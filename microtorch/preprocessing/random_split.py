from typing import Optional
import numpy as np
import math


def random_split(X: np.ndarray, y: np.ndarray, ratios: list[float], seed: Optional[int] = None):
    assert len(ratios) >= 2, "Ratios must have at least 2 elements"
    assert sum(ratios) == 1, "Sum of ratios must be equal to 1"
    assert len(X) == len(y), "X and y must have the same length"

    gen = np.random.default_rng(seed) if seed is not None else np.random

    n = len(X)
    shuffle = gen.permutation(n)
    X = X[shuffle]
    y = y[shuffle]

    split: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(len(ratios)):
        sz = math.ceil(ratios[i] * n)
        end = start + sz if i != len(ratios) - 1 else n
        split.append((
            X[start:end],
            y[start:end]
        ))
        start = end

    return split
