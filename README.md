## Perceptra

Perceptra is a Python project that implements a simple neural network (perceptrons) from scratch using NumPy. This project provides a comprehensive implementation of the neural network architecture, including forward and backward passes, activation functions, training, and defining the network architecture.

### Project Structure

The project contains the following files:

- `activations.py`: This file contains the implementation of various activation functions commonly used in neural networks, such as sigmoid, ReLU, and softmax.
- `training.py`: The `training.py` file includes the functions and logic for training the neural network using gradient descent and backpropagation algorithms.
- `forward_pass.py`: This file implements the forward pass functionality, where the input data propagates through the network layers, and the output is computed.
- `backward_pass.py`: The `backward_pass.py` file implements the backpropagation algorithm, computing the gradients of the loss function with respect to the network parameters.
- `architecture.py`: This file defines the neural network architecture, including the number of layers, number of neurons per layer, and the connections between layers.


### Getting Started

To get started with the Perceptra project, follow these steps:

1. Clone the repository: git clone https://github.com/khairiabidi/perceptra.git
2. Install the required dependencies: `pip install -r requirements.txt`
3. Explore the files in the project and familiarize yourself with the code structure.
4. Implement your own custom activation functions or modify existing ones in activations.py if needed.
5. Use the functions in forward_pass.py to perform forward propagation on your dataset.
6. Utilize the functions in backward_pass.py for backpropagation and gradient computation.
7. Use the training functions in training.py to train your neural network.
8. Experiment with different hyperparameters, activation functions, and architectures to improve your model's performance.
9. Enjoy using Perceptra for building and training your own simple neural networks!


### Contributin

Contributions to Perceptra are always welcome! If you have any suggestions, bug reports, or feature requests, please submit them to the GitHub repository at https://github.com/khairiabidi/perceptra. Contributions can be made through pull requests, addressing any identified issues or introducing new features.


### License

This project is licensed under the **MIT License**. Feel free to use and distribute this code for both commercial and non-commercial purposes.

### Acknowledgments

- The Perceptra project was inspired by the desire to understand and implement the fundamentals of neural networks using NumPy.
- Special thanks to the contributors of the NumPy library for providing a powerful mathematical framework for efficient array manipulation in Python.