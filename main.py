import numpy as np
from percepta.architecture import NeuralNetwork, LayerConfig
from percepta.activations import relu, sigmoid, relu_backward, sigmoid_backward
from percepta.training import train


def cli_main():
    epochs = 50
    learning_rate = 0.001

    x = np.random.randn(2, 100)
    y = np.random.randint(0, 2, (100, 1))

    layers = []
    layers.append(LayerConfig(input_dim=2, output_dim=5, activation=relu, activation_backward=relu_backward))
    layers.append(LayerConfig(input_dim=5, output_dim=5, activation=relu, activation_backward=relu_backward))
    layers.append(LayerConfig(input_dim=5, output_dim=5, activation=relu, activation_backward=relu_backward))
    layers.append(LayerConfig(input_dim=5, output_dim=5, activation=relu, activation_backward=relu_backward))
    layers.append(LayerConfig(input_dim=5, output_dim=1, activation=sigmoid, activation_backward=sigmoid_backward))

    neural_network = NeuralNetwork(layers_config=layers)

    neural_network, loss_history = train(x, y, neural_network, epochs, learning_rate)
    print(loss_history)


if __name__ == '__main__':
    cli_main()
