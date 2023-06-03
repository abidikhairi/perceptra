from typing import List, Callable

import numpy as np
from pydantic import BaseModel
from .activations import identity
from .initialization import random_gaussian_init


class LayerConfig(BaseModel):
    input_dim: int
    output_dim: int
    activation: Callable = identity
    activation_backward: Callable


class NeuralNetwork(BaseModel):
    layers_config: List[LayerConfig]
    weights: dict = {}


def init_neural_network(neural_network: NeuralNetwork, seed=1234) -> NeuralNetwork:
    np.random.seed(seed)
    weights = {}

    for idx, layer in enumerate(neural_network.layers_config):
        w = random_gaussian_init(layer.output_dim, layer.input_dim)
        b = random_gaussian_init(layer.output_dim, 1)

        weights[f'weights-{idx}'] = w
        weights[f'bias-{idx}'] = b

    neural_network.weights = weights
    return neural_network
