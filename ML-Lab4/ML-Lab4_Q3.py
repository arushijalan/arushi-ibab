import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# Load and inspect data
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("The contents of the data frame:")
    print(df.head())
    print("The shape of the data frame:")
    print(df.shape)
    print("The non-null values of the data frame:")
    print(df.info())
    print("The description of the data frame:")
    print(df.describe())
    print("The missing values in the data frame:")
    print(df.isnull().sum())
    return df


# Data standardization
def standardize_data(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_standardized = (x_train - mean) / std
    x_test_standardized = (x_test - mean) / std
    return x_train_standardized, x_test_standardized


# R^2 calculation
def r_squared(y_true, y_pred):
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


# Split data for training and testing
def split_data(df):
    # Select features and target
    X = df[["age"]]
    y = df["disease_score"]

    # # Handle categorical data (if Gender is categorical)
    # X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    # Standardize features
    X_train, X_test = standardize_data(X_train, X_test)

    # Add intercept term (column of ones) for both train and test sets
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    return X_train, X_test, y_train, y_test


# Hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)


# Cost function
def cost(X, y, theta):
    y_pred = hypothesis(X, theta)
    error = y - y_pred
    return np.sum(np.square(error)) / (2 * len(y))


# Gradient Descent for Linear Regression
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = -(1 / m) * np.dot(X.T, (y - hypothesis(X, theta)))
        theta = theta - alpha * gradient
        cost_history.append(cost(X, y, theta))
    return theta, cost_history



# Linear Regression using Normal Equation
def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Plotting the regression lines and actual data
def plot_results(X, y, y_pred_gd, y_pred_ne, y_pred_sklearn, feature_name):
    plt.figure(figsize=(10,6))
    X_feature = X[:, 1]  # Assuming the first column of X is the intercept, so select the second column (age)
    plt.scatter(X_feature, y, color='blue', label='Actual Data')  # Plotting actual data with feature "age"
    plt.plot(X_feature, y_pred_gd, color="orange", label="Gradient Descent Line", linewidth=2)
    plt.plot(X_feature, y_pred_ne, color='red', label='Normal Equation Line', linewidth=2)
    plt.plot(X_feature, y_pred_sklearn, color='green', label='Scikit-learn Line', linewidth=2, linestyle='dashed')

    plt.xlabel(feature_name)
    plt.ylabel('Target (Disease Score)')
    plt.title(f'Regression Line Comparison: {feature_name}')
    plt.legend()
    plt.show()


# Main function to train and evaluate using Sklearn, Gradient Descent, and Normal Equation
def main():
    file_path = '/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv'
    df = load_data(file_path)

    # Split data into train and test
    X_train, X_test, y_train, y_test = split_data(df)

    # Sklearn Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_sklearn = model.predict(X_test)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    print(f"R^2 from Sklearn Linear Regression: {r2_sklearn}")

    # Gradient Descent for Linear Regression
    initial_theta = np.zeros((X_train.shape[1], 1))
    theta_gd, cost_history = gradient_descent(X_train, y_train.values.reshape(-1, 1), initial_theta, alpha=0.001,iterations=5000)
    y_pred_gd = hypothesis(X_test, theta_gd).flatten()  # Flatten to 1D array
    r2_gd = r_squared(y_test, y_pred_gd)
    print(f"R^2 from Gradient Descent: {r2_gd}")

    # Normal Equation for Linear Regression
    theta_ne = normal_equation(X_train, y_train.values.reshape(-1, 1))
    y_pred_ne = hypothesis(X_test, theta_ne).flatten()  # Flatten to 1D array
    r2_ne = r_squared(y_test, y_pred_ne)
    print(f"R^2 from Normal Equation: {r2_ne}")

    # Plot results for comparison of methods
    feature_name = "Age"  # Change to any feature you want to plot against
    plot_results(X_test, y_test, y_pred_gd, y_pred_ne, y_pred_sklearn, feature_name)


if __name__ == "__main__":
    main()