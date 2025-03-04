#k-fold cross validation using SONAR dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

df=pd.read_csv("/home/ibab/sonar.csv")
print(df.head())
print(df.describe())
print(df.shape)
print(df.columns)

X=df.iloc[:,:-1]
y=df.iloc[:,-1]
LE=LabelEncoder()
y=LE.fit_transform(y)

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=999)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kf=KFold(n_splits=10,shuffle=True,random_state=999)

model= LogisticRegression(solver='liblinear',max_iter=1000,random_state=999)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
average_accuracy = np.mean(scores)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    print(f"  Training dataset index: {train_index}")
    print(f"  Test dataset index: {test_index}")

print(f"Accuracy Score for each fold: {[round(score, 10) for score in scores]}")
print(f"Average accuracy across 10 folds: {average_accuracy:.2f}")