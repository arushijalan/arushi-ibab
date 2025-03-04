# Implement information gain measures.
# The function should accept data points for parents,
# data points for both children and return an information gain value.

# import needed libraries
import numpy as np
from collections import Counter

# defining entropy using the formula
def entropy(labels):
    total = len(labels)
    label_counts = Counter(labels)
    entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values() if count > 0)
    return entropy_value

# defining information gain
def information_gain(parent_labels, left_child_labels, right_child_labels):
    total_parent = len(parent_labels)
    total_left = len(left_child_labels)
    total_right = len(right_child_labels)
    parent_entropy = entropy(parent_labels)
    left_entropy = entropy(left_child_labels)
    right_entropy = entropy(right_child_labels)
    weighted_child_entropy = (total_left / total_parent) * left_entropy + (total_right / total_parent) * right_entropy
    return parent_entropy - weighted_child_entropy
def main():
    # Example
    data_labels = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
    left_child = ["A", "A", "A", "C"]
    right_child = ["B", "B", "B", "C", "C", "C"]

    print("Entropy:", entropy(data_labels))
    print("Information Gain:", information_gain(data_labels, left_child, right_child))

if __name__ == "__main__":
    main()


