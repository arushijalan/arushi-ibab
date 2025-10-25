# Implement CNN using PyTorch for image classification using cifar10 dataset
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
# Plot train error vs increasing number of layers
# After some point, the training error increases with the number of layers

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def build_cnn_model(num_layers=2):

    layers = []

    in_channels = 3
    out_channels = 6
    for _ in range(num_layers):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        in_channels = out_channels
        out_channels *= 2

    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_channels * 5 * 5, 120))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(120, 84))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(84, 10))

    return nn.Sequential(*layers)


def train_model(model, train_loader, loss_fn, optimizer, device, num_epochs=20):
 
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_preds += (predictions.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_preds / total_samples
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch_losses


def test_model(model, test_loader, loss_fn, device):

    model.eval()
    total_loss = 0.0
    correct_preds = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
            correct_preds += (predictions.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct_preds / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


def visualize_images(images, labels, class_names):

    images = images / 2 + 0.5  # unnormalize
    np_images = images.numpy()
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.title("Sample CIFAR-10 Images")
    plt.show()
    print("Labels:", ' '.join(f'{class_names[labels[j]]:5s}' for j in range(4)))


def main():
    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load CIFAR-10 dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Display a few sample images
    data_iter = iter(train_loader)
    sample_images, sample_labels = next(data_iter)
    visualize_images(torchvision.utils.make_grid(sample_images[:4]), sample_labels[:4], class_labels)

    # Define loss and training setup
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 20
    learning_rate = 0.001

    # Compare models with different depths 
    layer_counts = [2, 3, 4, 5]
    avg_training_losses = []

    for layers in layer_counts:
        print(f"\nTraining CNN with {layers} convolutional layers...")
        model = build_cnn_model(num_layers=layers).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        epoch_losses = train_model(model, train_loader, loss_fn, optimizer, device, num_epochs)
        avg_training_losses.append(np.mean(epoch_losses))

    # Plot training error vs number of layers 
    plt.figure(figsize=(8, 6))
    plt.plot(layer_counts, avg_training_losses, marker='o', color='teal')
    plt.title("Training Error vs Number of CNN Layers")
    plt.xlabel("Number of Convolutional Layers")
    plt.ylabel("Average Training Loss")
    plt.grid(True)
    plt.show()

    # Evaluate final model 
    print("\nEvaluating the deepest model (5 layers) on test set:")
    final_model = build_cnn_model(num_layers=5).to(device)
    optimizer = optim.SGD(final_model.parameters(), lr=learning_rate, momentum=0.9)
    train_model(final_model, train_loader, loss_fn, optimizer, device, num_epochs)
    test_model(final_model, test_loader, loss_fn, device)


if __name__ == "__main__":
    main()
