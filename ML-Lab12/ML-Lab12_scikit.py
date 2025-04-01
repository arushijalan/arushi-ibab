# Constructing a Decision Tree for regression tasks using the scikit-learn library
# Utilizing the simulated dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define file path and load the dataset
file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
df = pd.read_csv(file_path)

# Define feature variables (X) and target variable (y)
# Exclude 'Gender', 'disease_score', and 'disease_score_fluct' columns from the features
X = df.drop(columns=["Gender", "disease_score", "disease_score_fluct"])
y = df["disease_score_fluct"]

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# Initialize and train the Decision Tree Regressor model
regressor = DecisionTreeRegressor(random_state=999)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Compute the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the structure of the trained Decision Tree model
plt.figure(figsize=(100, 10))
plot_tree(regressor, feature_names=X.columns)
plt.title("Decision Tree Regression Structure")
plt.show()
