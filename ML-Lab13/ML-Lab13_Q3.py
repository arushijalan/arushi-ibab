# Implement Random Forest algorithm for regression and classification using scikit-learn.
# Use diabetes and iris datasets.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import load_diabetes, load_iris

# Load Diabetes dataset for regression
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
# Split dataset into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=999)

# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=999)
rf_regressor.fit(X_train_reg, y_train_reg)  # Train the model
y_pred_reg = rf_regressor.predict(X_test_reg)  # Make predictions

# Evaluate model performance before K-Fold
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score before K-Fold: {r2}")

# K-Fold Cross-Validation for Regression
kf = KFold(n_splits=5, shuffle=True, random_state=999)
r2_scores_reg = []

for train_index, test_index in kf.split(X_reg):
    X_train_k, X_test_k = X_reg[train_index], X_reg[test_index]
    y_train_k, y_test_k = y_reg[train_index], y_reg[test_index]

    rf_regressor.fit(X_train_k, y_train_k)  # Train model on K-Fold split
    y_pred_k = rf_regressor.predict(X_test_k)  # Predict on test fold

    r2_scores_reg.append(r2_score(y_test_k, y_pred_k))  # Collect R2 scores

# Print average R2 score after K-Fold
print(f"R2 Score after K-Fold: {np.mean(r2_scores_reg)}\n")

# Load Iris dataset for classification
iris = load_iris()
X_clf, y_clf = iris.data, iris.target
# Split dataset into training and testing sets
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=999)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=999)
rf_classifier.fit(X_train_clf, y_train_clf)  # Train the model
y_pred_clf = rf_classifier.predict(X_test_clf)  # Make predictions

# Evaluate model performance before K-Fold
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Accuracy before K-Fold: {accuracy}")

# K-Fold Cross-Validation for Classification
kf = KFold(n_splits=5, shuffle=True, random_state=999)
accuracy_scores_clf = []

for train_index, test_index in kf.split(X_clf):
    X_train_k, X_test_k = X_clf[train_index], X_clf[test_index]
    y_train_k, y_test_k = y_clf[train_index], y_clf[test_index]

    rf_classifier.fit(X_train_k, y_train_k)  # Train model on K-Fold split
    y_pred_k = rf_classifier.predict(X_test_k)  # Predict on test fold

    accuracy_scores_clf.append(accuracy_score(y_test_k, y_pred_k))  # Collect accuracy scores

# Print average accuracy after K-Fold
print(f"Accuracy after K-Fold: {np.mean(accuracy_scores_clf)}")
