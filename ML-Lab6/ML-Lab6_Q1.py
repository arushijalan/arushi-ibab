# k-fold validation with scikit learn
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "/home/ibab/breast_cancer_data.csv"
data = pd.read_csv(file_path)

data = data.drop(columns=['id', "Unnamed: 32"])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
X = data.drop(columns=['diagnosis'])
y = data['diagnosis'].values

k_folds=10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=999)
model = LogisticRegression(max_iter=1000)
scores=[]
column_to_check='radius_mean'

for fold, (train_indices, test_indices) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    fold_score = np.mean(y_pred == y_test)
    scores.append(fold_score)

    column_mean_train = X_test_scaled[:, X.columns.get_loc(column_to_check)].mean()

    print(f"Fold {fold}:")
    print(f" Training Data Shape: {X_train_scaled.shape}")
    print(f" Test Data Shape: {X_test_scaled.shape}")
    print(f" Accuracy: {round(fold_score, 2)}\n")
    print(f" Mean of '{column_to_check}' in Test Data: {round(column_mean_train, 4)}\n")
mean_accuracy = np.mean(scores)
print(f"Average Accuracy: {mean_accuracy:.2f}")

# k_fold cross validation from scratch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

file_path = "/home/ibab/breast_cancer_data.csv"
data = pd.read_csv(file_path)

data = data.drop(columns=[col for col in ['id', 'Unnamed: 32'] if col in data.columns])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
X = data.drop(columns=['diagnosis'])
y = data['diagnosis'].values

k_fold=10
column_to_check='radius_mean'

def k_fold_indices(data, k_fold, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    fold_size=len(data)//k_fold
    folds=[]

    for i in range(k_fold):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        folds.append((train_indices, test_indices))
    return folds


def main():
    fold_indices = k_fold_indices(X, k_fold)
    model=LogisticRegression(max_iter=10000)
    scores=[]

    for fold,(train_indices, test_indices) in enumerate(fold_indices, start=1):
        X_train, y_train = X.iloc[train_indices], y[train_indices]
        X_test, y_test = X.iloc[test_indices], y[test_indices]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        fold_score = accuracy_score(y_test, y_pred)
        scores.append(fold_score)

        column_mean_train = X_test_scaled[:, X.columns.get_loc(column_to_check)].mean()
        print(f" Fold {fold}:")
        print(f" Training Data Shape: {X_train_scaled.shape}")
        print(f" Test Data Shape: {X_test_scaled.shape}")
        print(f" Accuracy: {round(fold_score, 2)}\n")
        print(f" Mean of '{column_to_check}' in Test Data: {round(column_mean_train, 4)}\n")
    mean_accuracy = np.mean(scores)
    print(f"Average Accuracy: {mean_accuracy:.2f}")

if __name__ == "__main__":
    main()