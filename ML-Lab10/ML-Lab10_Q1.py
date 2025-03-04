# import needed libraries
import numpy as np
from collections import Counter

# defining entropy
def entropy(labels):
    total = len(labels)
    label_counts = Counter(labels)
    entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values() if count > 0)
    return entropy_value

def main():
    # Example
    data_labels1 = ["A", "A", "B", "B", "B", "C", "C", "C", "C"]
    print("Entropy1:", entropy(data_labels1))
    data_labels2 = ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "C", "C"]
    print("Entropy2:", entropy(data_labels2))

if __name__ == "__main__":
    main()
