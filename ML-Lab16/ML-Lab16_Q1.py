# Python program to aggregate predictions from multiple trees to output a final
# prediction for a regression problem.

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Trains an ensemble of decision trees on bootstrapped subsets of the training data
def train_ensemble_trees(X, y, n_trees=10, max_depth=None):
    trees = []
    n_samples = X.shape[0]

    for _ in range(n_trees):
        # Create a bootstrap sample from the training data
        X_sample, y_sample = resample(X, y, n_samples=n_samples)
        
        # Train a decision tree on the sampled data
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X_sample, y_sample)
        trees.append(tree)

    return trees

# Aggregates predictions from all trees by averaging their outputs
def aggregate_predictions(trees, X):
    all_predictions = np.array([tree.predict(X) for tree in trees])  # Shape: (n_trees, n_samples)
    return np.mean(all_predictions, axis=0)

def main():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=999)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)

    # Train the ensemble of decision trees
    trees = train_ensemble_trees(X_train, y_train, n_trees=20, max_depth=5)

    # Generate predictions and evaluate performance
    y_pred = aggregate_predictions(trees, X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Ensemble Regressor - Mean Squared Error: {mse:.4f}")
    print(f"Ensemble Regressor - R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()
