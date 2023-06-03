import numpy as np

from .architecture import NeuralNetwork, init_neural_network
from .forward_pass import full_forward, get_cost_value
from .backward_pass import full_backward, update_parameters


def train(x: np.ndarray, y: np.ndarray, neural_network: NeuralNetwork, epochs: int, learning_rate: float = 0.001):
    neural_network = init_neural_network(neural_network, seed=1234)

    loss_history = []

    for _ in range(epochs):
        y_hat, memory = full_forward(x, neural_network)
        loss = get_cost_value(y_hat, y)
        loss_history.append(loss)

        grads_values = full_backward(y_hat, y, memory, neural_network)

        neural_network = update_parameters(grads_values, neural_network, learning_rate)

    return neural_network, loss_history