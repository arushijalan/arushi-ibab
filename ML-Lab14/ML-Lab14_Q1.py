# Implement Adaboost classifier using scikit-learn. Use the Iris dataset.
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# Initialize AdaBoost Classifier with Decision Tree as base estimator
adaboost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=999)
adaboost_clf.fit(X_train, y_train)  # Train the model

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
