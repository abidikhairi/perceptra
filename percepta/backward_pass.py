import numpy as np

from percepta.architecture import NeuralNetwork


def single_layer_backward(da_curr: np.ndarray, w_curr: np.ndarray, b_curr: np.ndarray,
                          z_curr: np.ndarray, a_prev: np.ndarray, activation_backward: callable):
    m = a_prev.shape[1]

    dz_curr = activation_backward(da_curr, z_curr)
    dw_curr = np.dot(dz_curr, a_prev.T) / m
    db_curr = np.sum(dz_curr, axis=1, keepdims=True) / m
    da_prev = np.dot(w_curr.T, dz_curr)

    return da_prev, dw_curr, db_curr


def full_backward(y_hat: np.ndarray, y: np.ndarray, memory: dict, neural_network: NeuralNetwork) -> dict:
    grads_values = {}
    m = y.shape[1]
    y = y.reshape(y_hat.shape)

    da_prev = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx, layer in reversed(list(enumerate(neural_network.layers_config))):
        backward_func = layer.activation_backward

        da_curr = da_prev

        a_prev = memory[f'activations-{layer_idx}']
        z_curr = memory[f'z-{layer_idx}']
        w_curr = neural_network.weights[f'weights-{layer_idx}']
        b_curr = neural_network.weights[f'bias-{layer_idx}']

        da_prev, dw_curr, db_curr = single_layer_backward(da_curr, w_curr, b_curr, z_curr, a_prev, backward_func)

        grads_values[f'dw-{layer_idx}'] = dw_curr
        grads_values[f'db-{layer_idx}'] = db_curr

    return grads_values


def update_parameters(grads_values: dict, neural_network: NeuralNetwork, learning_rate: float = 0.001) -> NeuralNetwork:
    for layer_idx, layer in enumerate(neural_network.layers_config):
        neural_network.weights[f'weights-{layer_idx}'] -= learning_rate * grads_values[f'dw-{layer_idx}']
        neural_network.weights[f'bias-{layer_idx}'] -= learning_rate * grads_values[f'db-{layer_idx}']

    return neural_network
