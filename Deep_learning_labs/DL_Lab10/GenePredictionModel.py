import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_prepare_data():
    data_matrix = np.load('1000G_reqnorm_float64.npy')
    num_genes, num_samples = data_matrix.shape
    print(f"Genes: {num_genes}, Samples: {num_samples}")

    # Z-score normalization
    gene_means = data_matrix.mean(axis=1, keepdims=True)
    gene_stds = data_matrix.std(axis=1, keepdims=True) + 1e-3
    data_matrix = (data_matrix - gene_means) / gene_stds

    num_landmark = 943
    X_data = data_matrix[:num_landmark, :].T  # (samples × landmark_genes)
    Y_data = data_matrix[num_landmark:, :].T  # (samples × target_genes)

    # Shuffle samples
    num_samples = X_data.shape[0]
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)

    X_data = X_data[shuffled_indices]
    Y_data = Y_data[shuffled_indices]

    # Define dimensions
    input_size = X_data.shape[1]
    output_size = Y_data.shape[1]

    # Split ratios
    train_end = int(0.7 * num_samples)
    val_end = int(0.9 * num_samples)

    X_train, Y_train = X_data[:train_end], Y_data[:train_end]
    X_val, Y_val = X_data[train_end:val_end], Y_data[train_end:val_end]
    X_test, Y_test = X_data[val_end:], Y_data[val_end:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    def make_dataset(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    return (
        make_dataset(X_train, Y_train),
        make_dataset(X_val, Y_val),
        make_dataset(X_test, Y_test),
        input_size,
        output_size
    )


def create_model(input_size, output_size, h1, h2, h3, dropout):
    layers = nn.Sequential(
        nn.Linear(input_size, h1),
        nn.Dropout(dropout),
        nn.BatchNorm1d(h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.Dropout(dropout),
        nn.BatchNorm1d(h2),
        nn.ReLU(),
        nn.Linear(h2, h3),
        nn.Dropout(dropout),
        nn.BatchNorm1d(h3),
        nn.ReLU(),
        nn.Linear(h3, output_size)
    )
    return layers


def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
    return epoch_loss / len(dataloader.dataset)


def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    # Load datasets
    train_data, val_data, test_data, input_size, output_size = load_and_prepare_data()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=32, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, drop_last=True)

    # Hyperparameter grid
    param_grid = {
        "h1": [800, 700],
        "h2": [600, 500],
        "h3": [500, 250],
        "dropout": [0.2, 0.3],
        "lr": [1e-3, 5e-4]
    }

    loss_fn = nn.MSELoss()
    best_config, best_val_loss = None, float("inf")

    # Hyperparameter tuning
    for h1 in param_grid["h1"]:
        for h2 in param_grid["h2"]:
            for h3 in param_grid["h3"]:
                for dr in param_grid["dropout"]:
                    for lr in param_grid["lr"]:
                        model = create_model(input_size, output_size, h1, h2, h3, dr).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

                        for _ in range(50):  # short tuning phase
                            train_one_epoch(train_loader, model, loss_fn, optimizer)

                        val_loss = evaluate_model(val_loader, model, loss_fn)
                        print(f"h1={h1}, h2={h2}, h3={h3}, dr={dr}, lr={lr} | Val Loss={val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_config = (h1, h2, h3, dr, lr)

    print(f"\nBest configuration: {best_config} | Validation Loss: {best_val_loss:.4f}")

    # Combine train + validation data for final training
    combined_train_loader = DataLoader(
        ConcatDataset([train_data, val_data]),
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    h1, h2, h3, dr, lr = best_config
    model = create_model(input_size, output_size, h1, h2, h3, dr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Train final model
    num_epochs = 1024
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(combined_train_loader, model, loss_fn, optimizer)
        val_loss = evaluate_model(val_loader, model, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Save trained model
    torch.save(model.state_dict(), "gene_expression_predictor.pth")
    print("Saved model as gene_expression_predictor.pth")

    # Evaluate on test set
    final_model = create_model(input_size, output_size, h1, h2, h3, dr).to(device)
    final_model.load_state_dict(torch.load("gene_expression_predictor.pth"))
    final_model.eval()

    test_loss = evaluate_model(test_loader, final_model, loss_fn)
    print(f"\nFinal Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
