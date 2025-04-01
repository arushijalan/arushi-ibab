## Implement decision tree classifier using scikit-learn using the sonar dataset.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load the sonar dataset
file_path = "sonar.csv"
df = pd.read_csv(file_path)

# Split dataset into features (X) and target variable (y)
X = df.iloc[:, :-1]  # Selecting all columns except the last one as features
y = df.iloc[:, -1]  # Selecting the last column as the target variable

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# Define and train the Decision Tree model
clf = DecisionTreeClassifier(random_state=999)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the decision tree
plt.figure(figsize=(10, 10))
plot_tree(clf, feature_names=X.columns, class_names=y.unique())
plt.show()
