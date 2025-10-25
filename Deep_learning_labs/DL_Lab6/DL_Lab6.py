import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def build_model():
    model_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model_layers


def train_loop(data_loader, model, loss_function, optimizer, device):
    dataset_size = len(data_loader.dataset)
    model.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            current = (batch_idx + 1) * len(inputs)
            print(f"Loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]")


def test_loop(data_loader, model, loss_function, device):
    size = len(data_loader.dataset)
    batches = len(data_loader)
    model.eval()
    total_loss, correct_predictions = 0, 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += loss_function(outputs, targets).item()
            correct_predictions += (outputs.argmax(1) == targets).type(torch.float).sum().item()

    avg_loss = total_loss / batches
    accuracy = 100 * correct_predictions / size
    print(f"Test Results → Accuracy: {accuracy:>0.1f}%, Avg Loss: {avg_loss:>8f}\n")


def visualize_predictions(model, dataset, device):

    labels_map = {
        0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
    }

    model.eval()
    fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        idx = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[idx]
        fig.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(image.squeeze(), cmap="gray")
    plt.show()


def main():
    # Load dataset
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())

    # Prepare data loaders
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Build model, define loss and optimizer
    model = build_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train and evaluate
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")
        train_loop(train_loader, model, loss_function, optimizer, device)
        test_loop(test_loader, model, loss_function, device)
    print("Training complete!")

    # Save model
    torch.save(model.state_dict(), "simple_ann.pth")
    print("Model saved as simple_ann.pth")

    # Load and test one sample
    model.load_state_dict(torch.load("simple_ann.pth"))
    model.eval()

    sample_img, sample_label = test_data[0]
    with torch.no_grad():
        sample_img = sample_img.to(device)
        prediction = model(sample_img.unsqueeze(0))
        predicted_label = prediction[0].argmax(0).item()
        print(f"Predicted: {predicted_label}, Actual: {sample_label}")

    # Visualization
    visualize_predictions(model, train_data, device)


if __name__ == "__main__":
    main()
