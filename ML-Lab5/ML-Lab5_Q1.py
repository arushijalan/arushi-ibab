from random import randint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


def data_load():
    simulated_data = pd.read_csv("/home/ibab/Machine_Learning_prac/ML_Lab4/housing.csv")
    X = simulated_data[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
    y=simulated_data["median_house_value"] #Ground Truth Values
    return (X,y)

def Scaled(X,y):
    X = X.fillna(0)
    y = y.fillna(0)
    means = X.mean(axis=0)
    std_devs = X.std(axis=0)
    X_scaled = (X - means) / std_devs
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std
    return X_scaled,y_scaled


def split_data(X_scaled,y_scaled):
    train=int(0.7*(X_scaled.shape[0]))
    X_train=X_scaled.iloc[:train]
    X_test=X_scaled.iloc[train:]
    y_train=y_scaled.iloc[:train]
    y_test=y_scaled.iloc[train:]
    return (X_train,X_test,y_train,y_test)


def initial_theta(X_train):
    n=X_train.shape[0]
    d=X_train.shape[1]
    Theta_values=[]
    for i in range(d):
        theta = 0
        Theta_values.append(theta)
    return Theta_values

def hypothesis_func(X_train,theta_Values):
    n = X_train.shape[0]
    d = X_train.shape[1]
    y1=[]
    for i in range (n):
        total = 0
        for j in range (d):
            value = X_train.iloc[i, j] * theta_Values[j]
            total += value
        y1.append(total)
    return (y1)

def Computing_error(y_train,y1):
    n = y_train.shape[0]
    error_list = []
    for i in range (n):
        Error = ( y1[i] - y_train.iloc[i] )
        error_list.append(Error)
    return (error_list)

def Computing_gradient(error_list,X_train):
    n = X_train.shape[0]
    d = X_train.shape[1]
    gradient=[0] * d

    idx = randint(0, n - 1)

    for i in range(d):
        gradient[i] = error_list[idx] * X_train.iloc[idx, i]

    return gradient
    # for i in range(d):
    #     value=0
    #     for j in range(n):
    #         value += error_list[j] * X_train.iloc[j,i]
    #     gradient.append(value)
    # return (gradient)

def updating_theta(gradient,theta_values):
    alpha = 0.001
    for i in range (len(theta_values)):
        theta_values[i] = theta_values[i] - alpha * gradient[i]
    return (theta_values)


def cost_func(error_list):
    total_error=sum(error**2 for error in error_list)
    cost_function = total_error/2
    return (cost_function)


def main():
    X,y = data_load()
    X_scaled , y_scaled = Scaled(X,y)
    X_train , X_test , y_train , y_test = split_data(X_scaled,y_scaled)
    print(X_train,X_test,y_train,y_test)
    theta_values = initial_theta(X_train)
    for i in range (1000):
        y1 = hypothesis_func(X_train,theta_values)
        error_list = Computing_error(y_train,y1)
        gradient = Computing_gradient(error_list,X_train)
        theta_values = updating_theta(gradient, theta_values)
        cost_function=cost_func(error_list)
        print(cost_function)


    y3 = hypothesis_func(X_test,theta_values)
    error_list = Computing_error(y_test,y3)
    cost_function=cost_func(error_list)
    print("Cost Function",cost_function)

    plt.scatter(y_test, y3)  # Provide x and y data directly
    plt.xlabel("True Values (Scaled)")  # Label for x-axis
    plt.ylabel("Predicted Values")  # Label for y-axis
    plt.title("True vs Predicted Values")  # Add a title
    plt.show()

    r2=r2_score(y_test,y3)
    print("R2 Score",r2)




if __name__ == '__main__':
    main()