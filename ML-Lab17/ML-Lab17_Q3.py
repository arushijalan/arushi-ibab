import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from mpl_toolkits.mplot3d import Axes3D


# Sample dataset for SVM
X_svm = np.array([
    [0.4, -0.7], [-1.5, -1.0], [-1.4, -0.9], [-1.3, -1.2],
    [-1.1, -0.2], [-1.2, -0.4], [-0.5, 1.2], [-1.5, 2.1],
    [1.0, 1.0], [1.3, 0.8], [1.2, 0.5], [0.2, -2.0],
    [0.5, -2.4], [0.2, -2.3], [0.0, -2.7], [1.3, 2.1],
])
y_svm = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

def plot_svm_decision_boundaries():
    """Plot SVM decision boundaries for different kernels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kernels = ["linear", "poly", "rbf"]

    for ax, kernel in zip(axes, kernels):
        clf = svm.SVC(kernel=kernel, gamma=2).fit(X_svm, y_svm)

        # Decision boundary and margins
        DecisionBoundaryDisplay.from_estimator(
            clf, X_svm, ax=ax, response_method="predict",
            plot_method="pcolormesh", alpha=0.3
        )
        DecisionBoundaryDisplay.from_estimator(
            clf, X_svm, ax=ax, response_method="decision_function",
            plot_method="contour", levels=[-1, 0, 1],
            colors=["k", "k", "k"], linestyles=["--", "-", "--"]
        )

        # Support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=150, facecolors="none", edgecolors="k", label="Support Vectors")

        # Plot original points
        scatter = ax.scatter(X_svm[:, 0], X_svm[:, 1], c=y_svm, s=30, edgecolors="k")
        ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.set_title(f"{kernel.capitalize()} Kernel")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    fig.suptitle("SVM Decision Boundaries with Linear, Polynomial, and RBF Kernels")
    plt.tight_layout()
    plt.show()

# ----- Logistic Regression with Feature Mapping -----

def transform_features(x1, x2):
    """Apply quadratic feature mapping."""
    return np.array([x1 ** 2, math.sqrt(2) * x1 * x2, x2 ** 2])

def plot_2d_points(x1, x2, labels):
    """Plot original 2D data points."""
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        color = "blue" if label == "Blue" else "red"
        plt.scatter(x1[i], x2[i], color=color,
                    label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original 2D Data Points")
    plt.legend()
    plt.show()

def plot_transformed_3d(x1_t, x2_t, x3_t, labels, model):
    """Visualize transformed 3D data and decision plane from logistic regression."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        color = "blue" if label == "Blue" else "red"
        ax.scatter(x1_t[i], x2_t[i], x3_t[i], color=color,
                   label=label if label not in ax.get_legend_handles_labels()[1] else "")

    ax.set_xlabel("x1 (original)")
    ax.set_ylabel("x2 (original)")
    ax.set_zlabel("x3 (transformed)")
    ax.set_title("3D Feature Mapped Data Points")
    ax.legend()

    # Decision plane
    xx, yy = np.meshgrid(
        np.linspace(min(x1_t), max(x1_t), 30),
        np.linspace(min(x2_t), max(x2_t), 30)
    )
    a, b, c = model.coef_[0]
    d = model.intercept_[0]
    zz = -(a * xx + b * yy + d) / c
    ax.plot_surface(xx, yy, zz, alpha=0.4, color='gray', edgecolor='none')

    plt.show()

def run_logistic_feature_mapping():
    """Run logistic regression on feature-mapped data and visualize in 3D."""
    raw_data = np.array([
        [1, 13, "Blue"], [1, 18, "Blue"], [2, 9, "Blue"], [3, 6, "Blue"],
        [6, 3, "Blue"], [9, 2, "Blue"], [13, 1, "Blue"], [18, 1, "Blue"],
        [3, 15, "Red"], [6, 6, "Red"], [6, 11, "Red"], [9, 5, "Red"],
        [10, 10, "Red"], [11, 5, "Red"], [12, 6, "Red"], [16, 3, "Red"]
    ])

    x1 = raw_data[:, 0].astype(float)
    x2 = raw_data[:, 1].astype(float)
    labels = raw_data[:, 2]
    y = np.array([1 if label == "Red" else 0 for label in labels])

    # Original 2D plot
    plot_2d_points(x1, x2, labels)

    # Transform features
    transformed = np.array([transform_features(x1[i], x2[i]) for i in range(len(x1))])
    x1_t, x2_t, x3_t = transformed[:, 0], transformed[:, 1], transformed[:, 2]

    # Logistic regression on transformed data
    model = LogisticRegression()
    model.fit(transformed, y)

    # Plot 3D decision boundary
    plot_transformed_3d(x1_t, x2_t, x3_t, labels, model)

def main():
    print("Plotting SVM decision boundaries...")
    plot_svm_decision_boundaries()
    
    print("\nRunning 3D feature transformation with logistic regression...")
    run_logistic_feature_mapping()

if __name__ == "__main__":
    main()
