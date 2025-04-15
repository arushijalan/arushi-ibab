# Implementation of XGBoost classifier and regressor using scikit-learn

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def load_datasets():
    # Load and split the Diabetes dataset for regression
    diabetes = load_diabetes()
    X_reg, y_reg = diabetes.data, diabetes.target
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=999
    )

    # Load and split the Iris dataset for classification
    iris = load_iris()
    X_cls, y_cls = iris.data, iris.target
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=999
    )

    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_cls, X_test_cls, y_train_cls, y_test_cls

def main():
    (
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        X_train_cls, X_test_cls, y_train_cls, y_test_cls
    ) = load_datasets()

    # Train the XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    xgb_regressor.fit(X_train_reg, y_train_reg)

    # Make predictions and evaluate regression performance
    y_pred_reg = xgb_regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"XGBoost Regressor - Mean Squared Error: {mse:.4f}")
    print(f"XGBoost Regressor - R² Score: {r2:.2f}")

    # Train the XGBoost Classifier
    xgb_classifier = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    xgb_classifier.fit(X_train_cls, y_train_cls)

    # Make predictions and evaluate classification performance
    y_pred_cls = xgb_classifier.predict(X_test_cls)
    accuracy = accuracy_score(y_test_cls, y_pred_cls)
    print(f"XGBoost Classifier - Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
