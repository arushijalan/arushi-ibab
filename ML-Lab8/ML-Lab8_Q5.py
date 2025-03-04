import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("/home/ibab/breast-cancer.csv", header=None)  # No column names in the dataset
print(df.head())
print(df.info())

X = df.iloc[:, :-1].astype(str) 
y = df.iloc[:, -1].astype(str) 

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

# Ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
X_train = ordinal_encoder.fit_transform(X_train)
X_test = ordinal_encoder.transform(X_test)

# Scale features (Logistic Regression benefits from scaling)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
