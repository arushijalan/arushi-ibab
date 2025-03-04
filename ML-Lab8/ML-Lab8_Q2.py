import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.linear_model import Ridge, Lasso, RidgeClassifier, LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

# Breast Cancer Dataset - Ridge & Lasso Regression
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge Regression
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print("R^2 Score for Ridge Regression (Breast Cancer):", r2_score(y_test, y_pred))

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("R^2 Score for Linear Regression (Breast Cancer):", r2_score(y_test, y_pred))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
print("R^2 Score for Lasso Regression (Breast Cancer):", r2_score(y_test, y_pred))


# Sonar Dataset - Logistic, Ridge, and Lasso Classification
df = pd.read_csv("/home/ibab/sonar.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

LE = LabelEncoder()
y = LE.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=1)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Logistic Regression Accuracy (Sonar):", accuracy_score(y_test, y_pred) * 100, "%")

# Ridge Classifier (Used for Classification)
ridge_clf = RidgeClassifier(alpha=6.0)
ridge_clf.fit(X_train, y_train)
y_pred = ridge_clf.predict(X_test)
print("Ridge Classifier Accuracy (Sonar):", accuracy_score(y_test, y_pred) * 100, "%")

# Lasso Regression (Thresholded to 0 or 1)
lasso = Lasso(alpha=0.0002)
lasso.fit(X_train, y_train)
y_pred = (lasso.predict(X_test) > 0.5).astype(int)
print("Lasso Regression Accuracy (Sonar):", accuracy_score(y_test, y_pred) * 100, "%")


# **California Housing Dataset - Ridge & Lasso Regression**
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge Regression
ridge = Ridge(alpha=5.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print("R^2 Score for Ridge Regression (California Housing):", r2_score(y_test, y_pred))

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("R^2 Score for Linear Regression (California Housing):", r2_score(y_test, y_pred))

# Lasso Regression
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
print("R^2 Score for Lasso Regression (California Housing):", r2_score(y_test, y_pred))
