import numpy as np


def random_gaussian_init(*args) -> np.ndarray:
    return np.random.randn(*args) * 0.1
