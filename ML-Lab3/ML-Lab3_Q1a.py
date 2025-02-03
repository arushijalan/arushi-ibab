import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
def load_data():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X = df[["age","BMI","BP","blood_sugar","Gender"]]
    y = df["disease_score"]
    return X,y
def main():
    [X,y] = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
    scaler = StandardScaler()
    scaler=scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    print("TRAINING")
    print("N=%d" %(len(X)))
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2=r2_score(y_test, y_pred)
    print("r2 score is  (close to 1 is good)" ,r2)
if __name__ == "__main__":
    main()