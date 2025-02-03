import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


def load_data():
    file_path = "breast_cancer_data.csv"
    data = pd.read_csv(file_path)
    print("Dataset:")
    print(data.head())

    data=data.drop(columns=['id',"Unnamed: 32"])
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    X= data.drop(columns=['diagnosis'])
    y = data['diagnosis'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_data(X_train_scaled, y_train):
    model=LogisticRegression(max_iter=10000)
    model.fit(X_train_scaled,y_train)
    return model


def predictions(model, X_test_scaled):
    y_pred = model.predict(X_test_scaled)
    return y_pred


def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)


def main():
    X_train_scaled, X_test_scaled, y_train, y_test = load_data()
    model = train_data(X_train_scaled, y_train)
    y_pred = predictions(model, X_test_scaled)
    evaluation(y_test, y_pred)

    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    plt.plot(y_pred,y_test)
    plt.show()


if __name__ == "__main__":
    main()