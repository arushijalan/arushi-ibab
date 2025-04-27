# 1. Develop prediction model for Iris.csv using joint probability distribution approach
# a. Use only the first two features, SepalLengthCm, SepalWidthCm and the target variable
# b. Add random noise to the features
# c. Discretize the feature values
# d. Build a decision tree model with max_depth = 2, then, compare the accuracy of this model with 
# the joint probability distribution method

# Import required libraries
import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris                 
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import KBinsDiscretizer      
from sklearn.tree import DecisionTreeClassifier         
from sklearn.metrics import accuracy_score              
from collections import defaultdict                    

def load_and_preprocess_data():
    iris = load_iris()

    # Use only the first two features: SepalLength and SepalWidth
    X = iris.data[:, :2]

    # Target variable (Species): values are 0 (setosa), 1 (versicolor), 2 (virginica)
    y = iris.target

    # Add Gaussian noise (mean=0, std=0.3) to simulate measurement errors
    np.random.seed(99)
    X += np.random.normal(0, 0.3, size=X.shape)

    # Discretize each feature into 3 equal-width bins
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X_discrete = discretizer.fit_transform(X)

    # Convert to DataFrame with readable column names
    X_discrete = pd.DataFrame(X_discrete, columns=['SepalLengthBin', 'SepalWidthBin'])

    # Split data into 70% training and 30% testing, stratified to keep class balance
    return train_test_split(X_discrete, y, test_size=0.3, random_state=999, stratify=y)

def train_joint_probability_model(X_train, y_train):
    # Dictionary to hold joint counts for feature combination and class label
    joint_counts = defaultdict(lambda: defaultdict(int))

    # Dictionary to hold total count of each class
    label_counts = defaultdict(int)

    # Count occurrences of each (feature_bin_1, feature_bin_2, label) combination
    for i in range(len(X_train)):
        x = tuple(X_train.iloc[i])  # Convert row to a tuple (e.g., (1.0, 2.0))
        label = y_train[i]
        joint_counts[x][label] += 1
        label_counts[label] += 1

    # Compute joint probabilities by dividing counts by total samples
    total = sum(label_counts.values())
    joint_probs = defaultdict(dict)
    for x_val, label_dict in joint_counts.items():
        for label in label_dict:
            joint_probs[x_val][label] = label_dict[label] / total

    return joint_probs, label_counts

def predict_with_joint_model(X_test, joint_probs, label_counts):
    predictions = []

    # Predict class for each test sample
    for i in range(len(X_test)):
        x = tuple(X_test.iloc[i])  # Feature bin combination
        label_scores = {}

        # Calculate joint probability P(x, y) for each class label
        for label in label_counts:
            label_scores[label] = joint_probs.get(x, {}).get(label, 0)

        # Predict label with the highest joint probability
        predicted_label = max(label_scores, key=label_scores.get)
        predictions.append(predicted_label)

    return predictions

def train_decision_tree(X_train, y_train):
    # Train a Decision Tree classifier with maximum depth 2
    clf = DecisionTreeClassifier(max_depth=2, random_state=999)
    clf.fit(X_train, y_train)
    return clf

def main():
    # Load, add noise, discretize, and split the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train joint model using frequency counts
    joint_probs, label_counts = train_joint_probability_model(X_train, y_train)

    # Predict and evaluate on test data
    y_pred_joint = predict_with_joint_model(X_test, joint_probs, label_counts)
    acc_joint = accuracy_score(y_test, y_pred_joint)

    # Train a decision tree and predict
    clf = train_decision_tree(X_train, y_train)
    y_pred_tree = clf.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)

    print(f"Accuracy (Joint Probability Model): {acc_joint:.2f}")
    print(f"Accuracy (Decision Tree, depth=2):  {acc_tree:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
