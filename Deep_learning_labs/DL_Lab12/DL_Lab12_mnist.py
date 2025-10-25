# Download MNIST dataset and implement a MNIST classifier using CNN PyTorch library.

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim


def build_cnn_model():
    cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return cnn_model


def train_model(model, training_loader, loss_function, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss_value = loss_function(predictions, labels)
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()
        print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {epoch_loss / len(training_loader):.4f}")


def evaluate_model(model, test_loader, loss_function, device):
    model.eval()
    total_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            test_loss = loss_function(predictions, labels)
            total_loss += test_loss.item()

            predicted_labels = predictions.argmax(dim=1, keepdim=True)
            total_correct += predicted_labels.eq(labels.view_as(predicted_labels)).sum().item()

    accuracy = 100 * total_correct / len(test_loader.dataset)
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformation
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root="data", train=True, download=False, transform=transform_pipeline)
    test_dataset = datasets.MNIST(root="data", train=False, download=False, transform=transform_pipeline)

    # Data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = build_cnn_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and evaluation
    train_model(model, train_loader, loss_function, optimizer, device, epochs=10)
    evaluate_model(model, test_loader, loss_function, device)


if __name__ == "__main__":
    main()
