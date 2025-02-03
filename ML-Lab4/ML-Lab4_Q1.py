import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
def load_data():
    cali_data = fetch_california_housing()
    X = cali_data.data  # Feature matrix
    y = cali_data.target  # Target variable (prices)
    print("Dataset Information:")
    print("Features:", cali_data.feature_names)
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    return X, y
def standardize_data(x_train, x_test):
    # Calculate the mean and standard deviation of the training data
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # Standardize the training data
    x_train_standardized = (x_train - mean) / std

    # Standardize the test data using the training set mean and std
    x_test_standardized = (x_test - mean) / std

    return x_train_standardized, x_test_standardized

def hypothesis(x_train,theta):
    return np.dot(x_train,theta)
def cost(x_train,y_train,theta):
    y_pred=hypothesis(x_train,theta)
    e=y_train - y_pred
    sq_error = np.square(e)
    j = np.sum(sq_error) / (2 * len(y_train))
    return j
def gradient(x_train,y_train, theta,alpha,iterations):
    n=len(y_train)
    cost_list=[]
    for i in range(iterations):
        y_pred=hypothesis(x_train,theta)
        e=y_train-y_pred
        g= -(1/n)*np.dot(x_train.T,e)
        theta=theta-alpha*g
        cost_list.append(cost(x_train,y_train,theta))
        if i == 1000 - 1:
            y_train_mean = np.mean(y_train)
            sst = np.sum((y_train - y_train_mean) ** 2)
            ssr = np.sum((y_pred - y_train_mean) ** 2)
            r2_sq = 1-(ssr / sst)
            print("r2_sq_score: ", r2_sq)
    return theta,cost_list

def update_theta(X,y,alpha=0.005,iterations=1000):
    # Split the data into train and test sets (80% training, 20% testing)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
    # Standardize the features
    x_train_standardized, x_test_standardized = standardize_data(x_train, x_test)

    # Add intercept term (column of ones) for both train and test sets
    x_train_standardized = np.c_[np.ones(x_train_standardized.shape[0]), x_train_standardized]
    x_test_standardized = np.c_[np.ones(x_test_standardized.shape[0]), x_test_standardized]

    theta = np.zeros((x_train_standardized.shape[1], 1))
    optimal_theta, cost_list = gradient(x_train_standardized, y_train.reshape(-1, 1), theta, alpha, iterations)
    print("Optimal Parameters (theta):", optimal_theta)
    print("Cost History:", cost_list)
    # Plot the cost history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cost_list, color='blue')
    plt.title('Cost History over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    return optimal_theta, cost_list
def main():
    X, y = load_data()
    optimal_theta, cost_list = update_theta(X, y)
if __name__ == '__main__':
    main()