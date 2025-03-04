from sklearn.datasets import load_breast_cancer
import numpy as np

X,y=load_breast_cancer(return_X_y=True)

def l2_norm(x):
    return np.sqrt(np.sum(x**2))

def l1_norm(x):
    return np.sum(np.abs(x))

l1_norms = np.apply_along_axis(l1_norm, axis=1, arr=X)
print("First 5 L1-norms:", l1_norms[:5])

l2_norms = np.apply_along_axis(l2_norm, axis=1, arr=X)
print("First 5 L2-norms:", l2_norms[:5])