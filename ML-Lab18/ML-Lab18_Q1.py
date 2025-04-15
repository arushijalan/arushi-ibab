import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import DecisionBoundaryDisplay


def load_data():
    #Prepare and encode the dataset consisting of x1, x2 features and class labels
    data = np.array([
        [6, 5, "Blue"], [6, 9, "Blue"], [8, 6, "Red"], [8, 8, "Red"],
        [8, 10, "Red"], [9, 2, "Blue"], [9, 5, "Red"], [10, 10, "Red"],
        [10, 13, "Blue"], [11, 5, "Red"], [11, 8, "Red"], [12, 6, "Red"],
        [12, 11, "Blue"], [13, 4, "Blue"], [14, 8, "Blue"]
    ])

    # Extract x1, x2 and label values
    x1 = data[:, 0].astype(float)
    x2 = data[:, 1].astype(float)
    label = data[:, 2]

    # Combine features for model training
    X = np.column_stack((x1, x2))

    # Convert string labels to numeric format (Blue -> 0, Red -> 1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(label)

    return X, y, encoder


def plot_svm_boundary(X, y, kernel_type, ax, title):
    #Train and visualize SVM decision boundaries for a given kernel
    model = svm.SVC(kernel=kernel_type, gamma='scale', degree=3, C=1)
    model.fit(X, y)

    # Display the decision boundary
    DecisionBoundaryDisplay.from_estimator(model,X,response_method="predict",plot_method="pcolormesh",alpha=0.3,ax=ax,)

    # Plot the training data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=80)
    ax.set_title(f"{title} Kernel")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(*scatter.legend_elements(), title="Class")


def main():
    # Load and preprocess the data
    X, y, encoder = load_data()

    # Create subplots for each kernel comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot decision boundary for RBF kernel
    plot_svm_boundary(X, y, kernel_type='rbf', ax=axes[0], title="RBF")

    # Plot decision boundary for Polynomial kernel
    plot_svm_boundary(X, y, kernel_type='poly', ax=axes[1], title="Polynomial")

    fig.suptitle("Comparison of SVM Decision Boundaries")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
