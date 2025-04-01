# Implement Gradient Boost Regression and Classification using scikit-learn.
# Use the Boston housing dataset from the ISLP package for the regression problem
# and weekly dataset from the ISLP package and
# use Direction as the target variable for the classification.

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, accuracy_score
from ISLP import load_data

# Load Boston Housing dataset for regression
boston = load_data('Boston')
X_boston = boston.drop(columns=['medv'])  # Features
y_boston = boston['medv']  # Target variable

# Split dataset
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_boston, y_boston, test_size=0.3, random_state=999)

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=999)
gbr.fit(X_train_b, y_train_b)  # Train model

y_pred_b = gbr.predict(X_test_b)  # Predict

# Evaluate model before K-Fold
r2_before = r2_score(y_test_b, y_pred_b)
print("Gradient Boosting Regression (Boston Housing Dataset)")
print(f"R2 Score before K-Fold: {r2_before}")

# K-Fold Cross-Validation for Regression
kf = KFold(n_splits=5, shuffle=True, random_state=999)
r2_scores = []

for train_index, test_index in kf.split(X_boston):
    X_train_k, X_test_k = X_boston.iloc[train_index], X_boston.iloc[test_index]
    y_train_k, y_test_k = y_boston.iloc[train_index], y_boston.iloc[test_index]

    gbr.fit(X_train_k, y_train_k)
    y_pred_k = gbr.predict(X_test_k)

    r2_scores.append(r2_score(y_test_k, y_pred_k))

print(f"R2 Score after K-Fold: {np.mean(r2_scores)}")

# Load Weekly dataset for classification
weekly = load_data('Weekly')
X_weekly = weekly.drop(columns=['Direction'])  # Features
y_weekly = (weekly['Direction'] == 'Up').astype(int)  # Convert Direction to binary target

# Split dataset
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_weekly, y_weekly, test_size=0.3, random_state=999)

# Initialize Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=999)
gbc.fit(X_train_w, y_train_w)  # Train model

y_pred_w = gbc.predict(X_test_w)  # Predict

# Evaluate model before K-Fold
accuracy_before = accuracy_score(y_test_w, y_pred_w)
print("Gradient Boosting Classification (Weekly Dataset)")
print(f"Accuracy before K-Fold: {accuracy_before}")

# K-Fold Cross-Validation for Classification
accuracy_scores = []

for train_index, test_index in kf.split(X_weekly):
    X_train_k, X_test_k = X_weekly.iloc[train_index], X_weekly.iloc[test_index]
    y_train_k, y_test_k = y_weekly.iloc[train_index], y_weekly.iloc[test_index]

    gbc.fit(X_train_k, y_train_k)
    y_pred_k = gbc.predict(X_test_k)

    accuracy_scores.append(accuracy_score(y_test_k, y_pred_k))

print(f"Accuracy after K-Fold: {np.mean(accuracy_scores)}")
