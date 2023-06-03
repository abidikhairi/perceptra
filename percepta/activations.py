import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.max(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def identity(x: np.ndarray) -> np.ndarray:
    return x


def sigmoid_backward(da: np.ndarray, z: np.ndarray):
    sig = sigmoid(z)
    return da * sig * (1 - sig)


def relu_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0

    return dz
