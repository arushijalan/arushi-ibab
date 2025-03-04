from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=999)

for col in range(X_train.shape[1]):
    X_train[:,col]=(X_train[:,col]-X_train[:,col].mean())/(X_train[:,col].max()-X_train[:,col].min())

for col in range(X_test.shape[1]):
    X_test[:,col]=(X_test[:,col]-X_test[:,col].mean())/(X_test[:,col].max()-X_test[:,col].min())

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression model accuracy by scratch(in %):", accuracy*100)