# Implement bagging regressor without using scikit-learn
import numpy as np
from sklearn.metrics import r2_score

class BaggingRegressor:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0):
        # Initialize the Bagging Regressor with the base model, number of estimators, and sample size
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        # Train multiple estimators on different bootstrap samples
        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples)
        self.models = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=sample_size, replace=True)  # Bootstrap sampling
            X_sample, y_sample = X[indices], y[indices]
            model = self._clone_estimator()
            model.fit(X_sample, y_sample)  # Train each estimator
            self.models.append(model)

    def predict(self, X):
        # Aggregate predictions from all trained estimators
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

    def _clone_estimator(self):
        # Create a new instance of the base model with the same parameters
        return self.base_estimator.__class__(**self.base_estimator.get_params())


# Example usage with Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Load diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# Train Bagging Regressor with Decision Tree as base estimator
base_model = DecisionTreeRegressor(max_depth=5)
bagging_regressor = BaggingRegressor(base_model, n_estimators=10, max_samples=0.8)
bagging_regressor.fit(X_train, y_train)

y_pred = bagging_regressor.predict(X_test)

# Evaluate performance using MSE and R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# K-Fold Cross-Validation to evaluate model stability
kf = KFold(n_splits=5, shuffle=True, random_state=999)
mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    bagging_regressor.fit(X_train, y_train)  # Train on each fold
    y_pred = bagging_regressor.predict(X_test)  # Predict on test fold

    mse_scores.append(mean_squared_error(y_test, y_pred))  # Collect MSE
    r2_scores.append(r2_score(y_test, y_pred))  # Collect R2 Score

# Print average performance across all folds
print(f"Mean Squared Error after K-Fold: {np.mean(mse_scores)}")
print(f"R2 Score after K-Fold: {np.mean(r2_scores)}")