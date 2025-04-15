import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_binary_iris_data():
    #Load the Iris dataset and filter to include only Setosa and Versicolor classes using first two features
    iris = datasets.load_iris()
    X = iris.data[:, :2]        # Use only the first two features
    y = iris.target
    feature_names = iris.feature_names[:2]

    # Select only class 0 and 1 (Setosa and Versicolor)
    binary_mask = y < 2
    return X[binary_mask], y[binary_mask], feature_names


def stratified_split(X, y, test_size=0.1):
    #Split the dataset while maintaining class balance
    X_train, X_test, y_train, y_test = [], [], [], []

    for label in np.unique(y):
        X_class = X[y == label]
        y_class = y[y == label]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_class, y_class, test_size=test_size, random_state=42
        )

        X_train.extend(X_tr)
        X_test.extend(X_te)
        y_train.extend(y_tr)
        y_test.extend(y_te)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def visualize_decision_boundary(model, X, y, feature_names, title="SVM Decision Boundary"):
    #Plot the SVM decision boundary along with data points
    h = 0.02  # Grid resolution

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    # Load and preprocess data
    X, y, feature_names = get_binary_iris_data()
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.1)

    # Train a linear SVM classifier
    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Evaluate model on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Visualize the decision boundary
    X_all = np.vstack((X_train, X_test))
    y_all = np.hstack((y_train, y_test))
    visualize_decision_boundary(model, X_all, y_all, feature_names)


if __name__ == "__main__":
    main()
