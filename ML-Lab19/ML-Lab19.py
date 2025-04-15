import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Function to compute confusion matrix values
def compute_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negative
    return tp, tn, fp, fn

# Accuracy = (TP + TN) / Total
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

# Precision = TP / (TP + FP)
def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

# Sensitivity / Recall = TP / (TP + FN)
def sensitivity(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

# Specificity = TN / (TN + FP)
def specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# F1-score = 2 * (Precision * Recall) / (Precision + Recall)
def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# Plot ROC curve and compute AUC
def plot_roc(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='pink', label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return roc_auc

def main():
    # Load dataset
    data = pd.read_csv('/home/ibab/datasets/heart.csv')

    # Determine the correct name for the target column
    if 'output' in data.columns:
        target_column = 'output'
    elif 'target' in data.columns:
        target_column = 'target'
    else:
        raise ValueError("Target column not found. Expected 'output' or 'target'.")

    # Separate features and label
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_probs = model.predict_proba(X_test)[:, 1]

    # Try different classification thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for t in thresholds:
        # Convert probabilities to binary predictions based on threshold
        y_pred = (y_probs >= t).astype(int)

        # Compute confusion matrix
        tp, tn, fp, fn = compute_confusion_matrix(y_test.values, y_pred)

        # Calculate all metrics
        acc = accuracy(tp, tn, fp, fn)
        prec = precision(tp, fp)
        sens = sensitivity(tp, fn)
        spec = specificity(tn, fp)
        f1 = f1_score(prec, sens)

        # Display results
        print(f"\nThreshold: {t}")
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Sensitivity (Recall): {sens:.4f}")
        print(f"Specificity: {spec:.4f}")
        print(f"F1-score: {f1:.4f}")

    # Plot ROC curve and show AUC
    auc_score = plot_roc(y_test.values, y_probs)
    print(f"\nAUC: {auc_score:.4f}")

# Execute main function
if __name__ == "__main__":
    main()
