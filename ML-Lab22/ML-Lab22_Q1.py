import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.cluster.hierarchy import linkage, fcluster
from ISLP import load_data


def load_nci_data():
    """
    Loads the NCI60 gene expression dataset.

    Returns:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target labels.
    """
    nci = load_data('NCI60')
    X = nci['data']
    y = nci['labels'].values.ravel()
    return X, y


def apply_pca(X, n_components=5):
    """
    Applies PCA for dimensionality reduction.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - n_components (int): Number of principal components.

    Returns:
    - X_reduced (np.ndarray): Reduced feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced


def apply_hierarchical_clustering(X, n_clusters=5):
    """
    Applies hierarchical clustering to reduce dimensionality.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - n_clusters (int): Number of feature clusters.

    Returns:
    - X_reduced (np.ndarray): Reduced feature matrix using cluster means.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    linkage_matrix = linkage(X_scaled.T, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    aggregated_features = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_mean = X_scaled[:, cluster_indices].mean(axis=1)
        aggregated_features.append(cluster_mean)

    X_reduced = np.vstack(aggregated_features).T
    return X_reduced


def evaluate_svm_model(X, y):
    """
    Trains an SVM classifier and evaluates train and test accuracy.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target labels.

    Returns:
    - train_acc (float): Training accuracy.
    - test_acc (float): Testing accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return train_acc, test_acc


def main():
    # Load gene expression data
    X, y = load_nci_data()

    # PCA
    X_pca = apply_pca(X, n_components=5)
    pca_train_acc, pca_test_acc = evaluate_svm_model(X_pca, y)
    print(f"PCA-based SVM Train Accuracy: {pca_train_acc:.4f}")
    print(f"PCA-based SVM Test Accuracy:  {pca_test_acc:.4f}")

    # Hierarchical Clustering
    X_hclust = apply_hierarchical_clustering(X, n_clusters=5)
    hclust_train_acc, hclust_test_acc = evaluate_svm_model(X_hclust, y)
    print(f"Hierarchical Clustering-based SVM Train Accuracy: {hclust_train_acc:.4f}")
    print(f"Hierarchical Clustering-based SVM Test Accuracy:  {hclust_test_acc:.4f}")

    # Visual comparison of test accuracy
    results_df = pd.DataFrame({
        "Method": ["PCA (5 PCs)", "Hierarchical Clustering (5 clusters)"],
        "Test Accuracy": [pca_test_acc, hclust_test_acc]
    })

    sns.barplot(data=results_df, x="Method", y="Test Accuracy")
    plt.ylim(0, 1)
    plt.title("SVM Classification Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
