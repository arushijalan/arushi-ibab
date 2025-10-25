# Write a program to simulate vanishing & exploding gradient problems.  
# https://machinelearningmastery.com/visualizing-the-vanishing-gradient-problem/ 
# Plot the gradient values to demonstrate the issues with training a very deep network

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

torch.manual_seed(99)
np.random.seed(99)


def build_network(activation_fn, hidden_units=5, total_layers=5):

    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh()
    }

    layers = [nn.Linear(2, hidden_units), activations[activation_fn]]

    for _ in range(total_layers - 2):
        layers.append(nn.Linear(hidden_units, hidden_units))
        layers.append(activations[activation_fn])

    layers.append(nn.Linear(hidden_units, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def compute_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        predictions = (y_pred.numpy() > 0.5).astype(int)
        acc = accuracy_score(y.numpy().reshape(-1), predictions)
    return acc


def record_weights(model):

    weights_snapshot = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights_snapshot[name] = param.detach().numpy().copy()
    return weights_snapshot


def record_gradients(model):

    grad_snapshot = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_snapshot[name] = param.grad.detach().numpy().copy()
    return grad_snapshot


def plot_weight_statistics(weight_records, epoch_list):

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), constrained_layout=True)

    axes[0].set_title("Mean of Weights")
    for key in weight_records[0]:
        axes[0].plot(epoch_list, [w[key].mean() for w in weight_records], label=key)
    axes[0].legend()

    axes[1].set_title("Standard Deviation of Weights")
    for key in weight_records[0]:
        axes[1].plot(epoch_list, [w[key].std() for w in weight_records], label=key)
    axes[1].legend()

    plt.show()


def plot_gradient_dynamics(grad_records, loss_records):

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 12), constrained_layout=True)

    axes[0].set_title("Mean of Gradients")
    for key in grad_records[0]:
        axes[0].plot(range(len(grad_records)), [g[key].mean() for g in grad_records], label=key)
    axes[0].legend()

    axes[1].set_title("S.D. of Gradients (Log Scale)")
    for key in grad_records[0]:
        axes[1].semilogy(range(len(grad_records)), [g[key].std() for g in grad_records], label=key)
    axes[1].legend()

    axes[2].set_title("Training Loss")
    axes[2].plot(range(len(loss_records)), loss_records)
    axes[2].set_xlabel("Epoch")

    plt.show()


def train_and_monitor(X, y, activation_fn="relu", num_epochs=100, batch_size=32):
    model = build_network(activation_fn=activation_fn, total_layers=5)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    weight_records = []
    grad_records = []
    epoch_list = []
    loss_records = []

    # Capture weights before training
    weight_records.append(record_weights(model))
    epoch_list.append(-1)

    print("Before training: Accuracy =", compute_accuracy(model, X, y))

    for epoch in range(num_epochs):
        model.train()
        indices = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            if i == 0:  # Record gradient for first batch each epoch
                grad_records.append(record_gradients(model))
                loss_records.append(loss.item())

        weight_records.append(record_weights(model))
        epoch_list.append(epoch)

    print("After training: Accuracy =", compute_accuracy(model, X, y))
    plot_weight_statistics(weight_records, epoch_list)

    return grad_records, loss_records


def main():

    # Generate two interleaving circles (non-linear classification)
    X_data, y_data = make_circles(n_samples=1000, factor=0.5, noise=0.1)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    # Visualize dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tensor[:, 0], X_tensor[:, 1], c=y_tensor.numpy().reshape(-1))
    plt.title("Two-Circle Dataset for Binary Classification")
    plt.show()

    # Compare different activation functions
    for act in ["sigmoid", "tanh", "relu"]:
        print(f"\nActivation Function: {act.upper()}")
        grad_records, loss_records = train_and_monitor(X_tensor, y_tensor, activation_fn=act)
        plot_gradient_dynamics(grad_records, loss_records)


if __name__ == "__main__":
    main()
