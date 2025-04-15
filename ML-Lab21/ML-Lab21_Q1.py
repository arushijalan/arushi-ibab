import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Function to compute Euclidean distance between points and a centroid
def compute_distance(point, centroid):
    return np.linalg.norm(point - centroid)

# Function to implement K-Means from scratch
def kmeans(X, k=3, max_iters=100, seed=42):
    np.random.seed(seed)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iters):
        # Step 1: Assign each point to the nearest centroid
        labels = np.array([np.argmin([compute_distance(x, c) for c in centroids]) for x in X])
        
        # Step 2: Recalculate centroids as mean of assigned points
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        
        # Step 3: Check for convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged after {i + 1} iterations.")
            break
        
        centroids = new_centroids

    return labels, centroids

def main():
    # Load Iris dataset and use only first two features for visualization
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length and sepal width

    # Run K-Means
    k = 3
    labels, centroids = kmeans(X, k)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
    plt.title("K-Means Clustering on Iris Dataset (from scratch)")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
