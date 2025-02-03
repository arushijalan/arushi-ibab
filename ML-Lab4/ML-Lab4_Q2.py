#using normal equation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.matrixlib.defmatrix import asmatrix
# Load the dataset
def load_data(df):
    print("The contents of the data frame is: ")
    print(df.head)
    print("The shape of the data frame is: ")
    print(df.shape)
    print("The non-null values of the data frame is: ")
    print(df.info())
    print("The description of the data frame is: ")
    print(df.describe())
    print("The missing values of the data frame is: ")
    print(df.isnull().sum())
    print("The column names of the data frame is: ")
    # print(df.columns)
    return df
def standardize_data(x_train, x_test):
    # Calculate the mean and standard deviation of the training data
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    # Standardize the training data
    x_train_standardized = (x_train - mean) / std
    # Standardize the test data using the training set mean and std
    x_test_standardized = (x_test - mean) / std
    return x_train_standardized, x_test_standardized
# Function to compute R^2 manually
def r_squared(y_true, y_pred):
    # Calculate Total Sum of Squares (TSS)
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    # Calculate Residual Sum of Squares (RSS)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    # R-squared formula
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2
#spliting of data
def split_data(df):
    data=load_data(df)
    s_data=data.sample(frac=1)
    train_size=0.70
    train_samples=int(len(s_data)*train_size)
    train_data=s_data.iloc[:train_samples]
    test_data=s_data.iloc[train_samples:]
    x_train=train_data.drop(["disease_score","disease_score_fluct","Gender"],axis=1)
    y_train = train_data[["disease_score"]]
    x_test=test_data.drop(["disease_score","disease_score_fluct","Gender"],axis=1)
    y_test = test_data[["disease_score"]]
      # Standardize the features (X_train and X_test)
    x_train_standardized, x_test_standardized = standardize_data(x_train.values, x_test.values)
    # Add intercept term (column of ones) for both train and test sets
    x_train_standardized = np.c_[np.ones(x_train_standardized.shape[0]), x_train_standardized]
    x_test_standardized = np.c_[np.ones(x_test_standardized.shape[0]), x_test_standardized]
    # Convert the data to matrices
    x_train_matrix = np.array(x_train_standardized)
    x_test_matrix = np.array(x_test_standardized)
    y_train_matrix = np.array(y_train.values).reshape(-1, 1)
    y_test_matrix = np.array(y_test.values).reshape(-1, 1)
    return x_train_matrix, x_test_matrix, y_train_matrix, y_test_matrix
def hypothesis(x_train,theta):
    return np.dot(x_train,theta)
def cost(x_train,y_train,theta):
    y_pred=hypothesis(x_train,theta)
    e=y_train - y_pred
    sq_error = np.square(e)
    j = np.sum(sq_error) / (2 * len(y_train))
    return j
def theta_cal(x_train,y_train):
    theta=np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    cost_list = []
    cost_list.append(cost(x_train, y_train, theta))
    return theta, cost_list
def calcul_r(f):
    x_train, x_test, y_train, y_test = split_data(f)
    theta, cost_list = theta_cal(x_train, y_train)
    y_pred = hypothesis(x_test,theta)
    r2 = r_squared(y_test, y_pred)
    print("R^2 (Coefficient of Determination):", r2)
    print("Optimal Parameters (theta):", theta)
    print("Cost History:", cost_list)
    # Plot the cost history
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(cost_list, color='blue')
    # plt.title('Cost History over Theta')
    # plt.xlabel('Theta')
    # plt.ylabel('Cost')
    # plt.show()
    return theta, cost_list
def main():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    theta, cost_list = calcul_r(df)
    # print(cost_list[-1]<cost_list[0])
    # print(cost_list[-1])
    # print(cost_list[0])
if __name__ == '__main__':
    main()