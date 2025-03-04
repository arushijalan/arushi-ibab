import numpy as np

class OrdinalEncoder:
    def __init__(self):
        self.mapping = {}
    
    def fit(self, data):
        unique_classes = np.unique(data)
        self.mapping = {category: idx for idx, category in enumerate(unique_classes)}
    
    def transform(self, data):
        return np.array([self.mapping.get(val, -1) for val in data]) 
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class OneHotEncoder:
    def __init__(self):
        self.categories = []
    
    def fit(self, data):
        self.categories = np.unique(data)
    
    def transform(self, data):
        num_categories = len(self.categories)
        one_hot_matrix = np.zeros((len(data), num_categories))
        for i, val in enumerate(data):
            category_idx = np.where(self.categories == val)[0][0]
            one_hot_matrix[i, category_idx] = 1
        return one_hot_matrix
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

data = np.array(["Red", "Green", "Blue", "Green", "Red", "Blue", "Red"])

ordinal_encoder = OrdinalEncoder()
ordinal_encoded = ordinal_encoder.fit_transform(data)
print("Ordinal Encoded Data:", ordinal_encoded)

one_hot_encoder = OneHotEncoder()
one_hot_encoded = one_hot_encoder.fit_transform(data)
print("One-Hot Encoded Data:\n", one_hot_encoded)
