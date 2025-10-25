# 1. Consider two networks.  
# W is a matrix, x is a vector, z is a vector, and a is a vector. 
# y^ is a scalar and a final prediction. Initialize x, w randomly, 
# z is a dot product of x and w, a is ReLU(z).  Initialize X and W randomly. 
# Every neuron has a bias term. 

# 2. Implement forward pass for the above two networks. 
# Print activation values for each neuron at each layer. 
# Print the loss value (y^).

# 3.  Implement the forward pass using vectorized operations, 
# i.e. W should be a matrix, x, z and a are vectors.  
# The implementation should not contain any loops. 

import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

def forward_pass(layers, x, weights, biases):
    activations = [x]  # input is the first activation
    a = x

    for i in range(1, len(layers)):
        z = np.dot(weights[i-1], a) + biases[i-1]

        if i < len(layers) - 1:  
            a = relu(z)
        else:
            a = softmax(z)

        print(f"\nLayer {i}:")
        print("z:", z)
        print("a:", a)

        activations.append(a)

    return activations

def compute_loss(y_hat, y_true):
    return 0.5 * np.sum((y_true - y_hat) ** 2)

def main():
    # Number of layers
    num_layers = int(input("Enter total number of layers (including input and output): "))

    # Neurons in each layer
    layers = list(map(int, input(f"Enter number of neurons in each of the {num_layers} layers (space-separated): ").split()))
    print("Network architecture:", layers)

    # Input values
    x = np.array(list(map(float, input(f"Enter {layers[0]} input values (space-separated): ").split())))

    # Collect weights and biases
    weights, biases = [], []
    for i in range(1, num_layers):
        num_inputs, num_outputs = layers[i-1], layers[i]
        print(f"\nLayer {i}: Expecting {num_outputs} neurons, each with {num_inputs} weights.")

        w_vals = list(map(float, input(f"Enter {num_outputs * num_inputs} weights (space-separated): ").split()))
        W = np.array(w_vals).reshape(num_outputs, num_inputs)
        weights.append(W)

        b_vals = list(map(float, input(f"Enter {num_outputs} biases (space-separated): ").split()))
        b = np.array(b_vals)
        biases.append(b)

    # Forward pass
    activations = forward_pass(layers, x, weights, biases)
    y_hat = activations[-1]
    print("\nFinal prediction (y^):", y_hat)

    # Loss
    y_true = np.array(list(map(float, input(f"\nEnter {layers[-1]} true target values (space-separated): ").split())))
    loss = compute_loss(y_hat, y_true)
    print("Loss:", loss)

if __name__ == "__main__":
    main()
