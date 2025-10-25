# 1 Implement a 1-layer (input - output layer) neural network from scratch for the following dataset. 
# 2 This includes implementing forward and backward passes from scratch. 
# 3 Print the training loss and plot it over 1000 iterations. 
# 4 Visualize at the weights and gradients of the layers - https://machinelearningmastery.com/visualizing-the-vanishing-gradient-problem/ 
# 5 https://wandb.ai/ayush-thakur/debug-neural-nets/reports/Visualizing-and-Debugging-Neural-Networks-with-PyTorch-and-Weights-Biases--Vmlldzo2OTUzNA
# 6 https://karpathy.github.io/2019/04/25/recipe/


import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_error_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def main():
    inputs = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)

    targets = np.array([[0], [1], [1], [0]], dtype=float)

    # Hyperparameters
    np.random.seed(42)
    learning_rate = 0.1
    num_epochs = 1000

    # Network dimensions 
    num_inputs = inputs.shape[1]
    num_outputs = 1

    # Weight and bias initialization 
    weights = np.random.randn(num_inputs, num_outputs)
    bias = np.zeros((1, num_outputs))

    # Tracking metrics 
    loss_history = []
    weight_mean_history = []
    grad_norm_history = []

    # Training 
    for epoch in range(num_epochs):
        # Forward pass 
        z = np.dot(inputs, weights) + bias
        predictions = sigmoid(z)

        # Compute loss
        loss = mean_squared_error(targets, predictions)
        loss_history.append(loss)

        # Backward pass
        grad_loss_output = mean_squared_error_grad(targets, predictions)
        grad_output_z = sigmoid_derivative(z)
        grad_loss_z = grad_loss_output * grad_output_z

        grad_weights = np.dot(inputs.T, grad_loss_z)
        grad_bias = np.sum(grad_loss_z, axis=0, keepdims=True)

        # Parameter update 
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        # Track weights and gradients
        weight_mean_history.append(np.mean(weights))
        grad_norm_history.append(np.linalg.norm(grad_weights))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d} | Loss: {loss:.6f}")

    # Final predictions 
    print("\nFinal predictions after training:")
    print(predictions)

    # Plot: Training Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label="Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)

    # Plot: Mean Weight Evolution 
    plt.subplot(1, 3, 2)
    plt.plot(weight_mean_history, label="Mean Weight", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Weight Value")
    plt.title("Weight Evolution")
    plt.grid(True)

    # Plot: Gradient Norm Evolution 
    plt.subplot(1, 3, 3)
    plt.plot(grad_norm_history, label="Gradient Norm", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Magnitude Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
