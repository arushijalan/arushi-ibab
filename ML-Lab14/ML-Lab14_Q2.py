# Implement Adaboost classifier without using scikit-learn. Use the Iris dataset.

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

class CustomAdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples  # Initialize weights equally

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)  # Weak learner
            model.fit(X, y, sample_weight=weights)  # Train model with weighted samples
            y_pred = model.predict(X)

            error = np.sum(weights * (y_pred != y)) / np.sum(weights)  # Weighted error rate
            if error > 0.5:
                continue  # Skip if error is too high

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # Compute model weight
            weights *= np.exp(-alpha * y * y_pred)  # Update sample weights
            weights /= np.sum(weights)  # Normalize weights

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)
        return np.sign(final_pred)  # Use sign function for final prediction


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to binary classification (for simplicity)
y = np.where(y == 2, 1, -1)  # Convert labels to -1 and 1

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# Initialize and train Custom AdaBoost classifier
adaboost_clf = CustomAdaBoost(n_estimators=50)
adaboost_clf.fit(X_train, y_train)

y_pred = adaboost_clf.predict(X_test)  # Make predictions

# Evaluate model performance before K-Fold
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy before K-Fold: {accuracy}")

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=999)
accuracy_scores = []

for train_index, test_index in kf.split(X):
    X_train_k, X_test_k = X[train_index], X[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]

    adaboost_clf.fit(X_train_k, y_train_k)  # Train model on K-Fold split
    y_pred_k = adaboost_clf.predict(X_test_k)  # Predict on test fold

    accuracy_scores.append(accuracy_score(y_test_k, y_pred_k))  # Collect accuracy scores

# Print average accuracy after K-Fold
print(f"Accuracy after K-Fold: {np.mean(accuracy_scores)}")