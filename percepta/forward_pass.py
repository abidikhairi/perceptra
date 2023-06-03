import numpy as np
from .architecture import NeuralNetwork


def single_layer_forward(x_prev: np.ndarray, w: np.ndarray, b: np.ndarray, activation: callable) -> (np.ndarray, np.ndarray):
    z = np.dot(w, x_prev) + b

    return activation(z), z


def full_forward(x: np.ndarray, neural_network: NeuralNetwork) -> (np.ndarray, dict):
    memory = {}
    weights = neural_network.weights
    x_curr = x

    for layer_idx, layer in enumerate(neural_network.layers_config):
        activation = layer.activation
        w = weights[f'weights-{layer_idx}']
        b = weights[f'bias-{layer_idx}']

        a_prev = x_curr

        a_curr, z_curr = single_layer_forward(a_prev, w, b, activation)

        memory[f"activations-{layer_idx}"] = a_prev
        memory[f"z-{layer_idx}"] = z_curr

    return a_curr, memory


def get_cost_value(y_hat: np.ndarray, y: np.ndarray):
    m = y_hat.shape[1]
    cost = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))

    return np.squeeze(cost)