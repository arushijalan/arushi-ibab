#without normalization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

df=pd.read_csv("/home/ibab/sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
LE=LabelEncoder()
y=LE.fit_transform(y)

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
model= LogisticRegression(solver='liblinear',max_iter=1000,random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)

print("Logistic Regression model accuracy without normalization(in %):", accuracy*100)

#with normalization(sklearn implementation)
df=pd.read_csv("/home/ibab/Downloads/sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
LE=LabelEncoder()
y=LE.fit_transform(y)

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

model= LogisticRegression(solver='liblinear',max_iter=1000,random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc=accuracy_score(y_test, y_pred)

print("Logistic Regression model accuracy with normalization using sklearn(in %):", acc*100)

#with normalization (from scratch)
df=pd.read_csv("/home/ibab/Downloads/sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
LE=LabelEncoder()
y=LE.fit_transform(y)

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=999)

for col in range(X_train.shape[1]):
    X_train.iloc[:,col]=(X_train.iloc[:,col]-X_train.iloc[:,col].mean())/(X_train.iloc[:,col].max()-X_train.iloc[:,col].min())

for col in range(X_test.shape[1]):
    X_test.iloc[:,col]=(X_test.iloc[:,col]-X_test.iloc[:,col].mean())/(X_test.iloc[:,col].max()-X_test.iloc[:,col].min())

model= LogisticRegression(solver='liblinear',max_iter=1000,random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)

print("Logistic Regression model accuracy with normalization from scratch(in %):", accuracy*100)