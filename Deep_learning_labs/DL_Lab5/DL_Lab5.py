# Implement a 2-layer (input layer, hidden layer and output layer) neural network from scratch for the XOR operation. 
# This includes implementing forward and backward passes from scratch. 

import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)


def forward_pass(input_vector, hidden_weights, hidden_biases, output_weights, output_bias):
    input_vector = np.array(input_vector)

    # Hidden layer computation
    for i in range(len(hidden_weights)):
        z_hidden = np.dot(hidden_weights[i], input_vector) + hidden_biases[i]
        hidden_output = relu(z_hidden)
        input_vector = hidden_output  # Pass to next layer

    # Output layer computation
    final_output = np.dot(output_weights, hidden_output) + output_bias
    return relu(final_output), hidden_output


def main():
    # Input data for XOR 
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Initialize network parameters
    hidden_weights = [np.array([[1, 1], [1, 1]])]
    hidden_biases = [np.array([0, -1])]

    output_weights = np.array([[1, -2]])
    output_bias = np.array([0])

    hidden_layer_x = []
    hidden_layer_y = []

    # Forward pass for all inputs
    for sample in input_data:
        output, hidden_output = forward_pass(sample, hidden_weights, hidden_biases, output_weights, output_bias)
        print(f"Input: {sample}, Output: {output}")
        hidden_layer_x.append(hidden_output[0])
        hidden_layer_y.append(hidden_output[1])

    # Visualization of hidden layer activations
    plt.grid(True)
    plt.scatter(hidden_layer_x, hidden_layer_y, color="purple")
    plt.xlabel("Hidden Neuron 1 Activation")
    plt.ylabel("Hidden Neuron 2 Activation")
    plt.title("Hidden Layer Activation Space (XOR)")
    plt.show()


if __name__ == '__main__':
    main()
