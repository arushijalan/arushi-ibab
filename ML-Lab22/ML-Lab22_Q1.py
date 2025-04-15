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
    # Load NCI60 gene expression dataset
    nci = load_data('NCI60')
    X = nci['data']
    y = nci['labels'].values.ravel()
    return X, y


def apply_pca(X, n_components=5):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced


def apply_hierarchical_clustering(X, n_clusters=5):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform hierarchical clustering on the transposed data (features)
    linkage_matrix = linkage(X_scaled.T, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Aggregate features by cluster mean
    aggregated_features = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_mean = X_scaled[:, cluster_indices].mean(axis=1)
        aggregated_features.append(cluster_mean)

    X_reduced = np.vstack(aggregated_features).T
    return X_reduced


def evaluate_svm_model(X, y):
    # Train and evaluate SVM model using accuracy
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc


def main():
    # Load gene expression data
    X, y = load_nci_data()

    # Reduce dimensions using PCA and evaluate
    X_pca = apply_pca(X, n_components=5)
    pca_acc = evaluate_svm_model(X_pca, y)
    print(f"PCA-based SVM Accuracy: {pca_acc:.4f}")

    # Reduce dimensions using hierarchical clustering and evaluate
    X_hclust = apply_hierarchical_clustering(X, n_clusters=5)
    hclust_acc = evaluate_svm_model(X_hclust, y)
    print(f"Hierarchical Clustering-based SVM Accuracy: {hclust_acc:.4f}")

    # Compare accuracies visually
    results_df = pd.DataFrame({
        "Method": ["PCA (5 PCs)", "Hierarchical Clustering (5 clusters)"],
        "Accuracy": [pca_acc, hclust_acc]
    })

    sns.barplot(data=results_df, x="Method", y="Accuracy")
    plt.ylim(0, 1)
    plt.title("SVM Classification Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
