from typing import Optional
import numpy as np
import math


def random_split(*arrays: np.ndarray, ratios: list[float], seed: Optional[int] = None):
    assert len(arrays) > 0, "At least one array must be provided"
    assert len(ratios) >= 2, "Ratios must have at least 2 elements"
    assert sum(ratios) == 1, "Sum of ratios must be equal to 1"
    
    n = len(arrays[0])
    for arr in arrays:
        assert len(arr) == n, "All arrays must have the same length"

    gen = np.random.default_rng(seed) if seed is not None else np.random

    shuffle = gen.permutation(n)
    shuffled_arrays = [arr[shuffle] for arr in arrays]

    split: list[list[np.ndarray]] = []
    start = 0
    for i in range(len(ratios)):
        sz = math.ceil(ratios[i] * n)
        end = start + sz if i != len(ratios) - 1 else n
        split.append([arr[start:end] for arr in shuffled_arrays])
        start = end

    return split
