import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ISLP import load_data 
from statsmodels.datasets import get_rdataset


def biplot(score, coeff, labels=None, states=None):
    # Extract principal component scores
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    n = coeff.shape[0]

    # Create 3D plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=5)

    # Add state labels to each point
    if states is not None:
        for i in range(len(xs)):
            ax.text(xs[i], ys[i], zs[i], states[i], size=7)

    # Plot arrows for each variable
    for i in range(n):
        ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)
        label = labels[i] if labels is not None else f"Var{i+1}"
        ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i, 2] * 1.15, label, color='g', ha='center', va='center')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D Biplot")
    plt.grid()
    plt.show()


def main():
    # Load USArrests dataset
    df = get_rdataset('USArrests').data
    X = df.values
    states = df.index
    labels = df.columns

    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_std)

    # Display PCA statistics
    df_pca = pd.DataFrame(X_pca)
    std_dev = df_pca.describe().transpose()["std"]
    print(f"Standard deviation: {std_dev.values}")
    print(f"Proportion of Variance Explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative Proportion: {np.cumsum(pca.explained_variance_)}")

    # Plot 3D biplot using the first 3 principal components
    biplot(X_pca[:, :3], pca.components_[:3, :].T, labels=list(labels), states=states)

    # Calculate feature importance for PC1, PC2, and PC3
    pc1 = abs(pca.components_[0])
    pc2 = abs(pca.components_[1])
    pc3 = abs(pca.components_[2])

    feature_importance = pd.DataFrame({
        "Features": list(labels),
        "PC1 Importance": pc1,
        "PC2 Importance": pc2,
        "PC3 Importance": pc3
    })
    print(feature_importance)

    # Plot cumulative explained variance
    plt.figure()
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_),
             color='red')
    plt.xlabel('Components')
    plt.ylabel('Explained variance')
    plt.title("Cumulative Explained Variance")
    plt.grid()
    plt.show()

    # Plot scree plot
    plt.figure()
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance ratio')
    plt.title("Scree Plot")
    plt.grid()
    plt.show()

    # Create new DataFrame with top 3 PCA components
    pca_df = pd.DataFrame(X_pca[:, :3], index=df.index)
    print(pca_df.head())


if __name__ == "__main__":
    main()
