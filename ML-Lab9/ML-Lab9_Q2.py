import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
file_path = "/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=["Gender", "disease_score", "disease_score_fluct"])
y = df["disease_score_fluct"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
regressor = DecisionTreeRegressor(random_state=999)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
plt.figure(figsize=(100, 10))
plot_tree(regressor, feature_names=X.columns)
plt.title("Decision Tree Regression Structure")
plt.show()
