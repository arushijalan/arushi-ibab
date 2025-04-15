# Let x1 = [3, 6], x2 = [10, 10].
# Use the above “Transform” function to transform these vectors
# to a higher dimension and  compute the dot product in a higher dimension.
# Print the value.

import numpy as np
import math

def transform(a, b):
    #Transforms two input values into a higher-dimensional space
    return np.array([
        a ** 2,
        math.sqrt(2) * a * b,
        b ** 2
    ])

def main():
    # Original vectors
    x1 = np.array([3, 6])
    x2 = np.array([10, 10])

    # Apply the transformation
    phi_1 = transform(x1[0], x2[0])
    phi_2 = transform(x1[1], x2[1])

    print("Transformed vector 1:", phi_1)
    print("Transformed vector 2:", phi_2)

    # Compute and print the dot product in higher-dimensional space
    dot_product = np.dot(phi_1, phi_2)
    print("Dot product in higher-dimensional space:", dot_product)

if __name__ == "__main__":
    main()
