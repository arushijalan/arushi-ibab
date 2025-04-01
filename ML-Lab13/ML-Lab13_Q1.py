# Implement bagging regressor and classifier using scikit-learn.
# Use diabetes and iris datasets.

# importing needed libraries and functions
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import numpy as np

## For Diabetes dataset
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.3, random_state=999)

# Bagging Regressor
regressor = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100, random_state=999)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluating Regression Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Bagging Regressor MSE: {mse:.4f}")
print(f"Bagging Regressor R2 Score: {r2:.4f}")

# K-Fold Cross Validation for Regression
kf = KFold(n_splits=5, shuffle=True, random_state=999)
mse_scores = -cross_val_score(regressor, diabetes.data, diabetes.target, cv=kf, scoring='neg_mean_squared_error')
print(f"Mean MSE across K-Folds: {np.mean(mse_scores):.4f}")
r2_scores = cross_val_score(regressor, diabetes.data, diabetes.target, cv=kf, scoring='r2')
print(f"Mean R2 Score across K-Folds: {np.mean(r2_scores):.4f}")

## For Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=999)

# Bagging Classifier
classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=999)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluating Classification Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Classifier Accuracy: {accuracy:.4f}")
r2 = r2_score(y_test, y_pred)
print(f"Bagging Classifier R2 Score: {r2:.4f}")

# K-Fold Cross Validation for Classification
accuracy_scores = cross_val_score(classifier, iris.data, iris.target, cv=kf, scoring='accuracy')
print(f"Mean Accuracy across K-Folds: {np.mean(accuracy_scores):.4f}")
r2_scores = cross_val_score(classifier, iris.data, iris.target, cv=kf, scoring='r2')
print(f"Mean R2 Score across K-Folds: {np.mean(r2_scores):.4f}")

