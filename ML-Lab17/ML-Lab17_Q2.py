#  Implement a polynomial kernel
#  K(a,b) =  a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2 .
#  Apply this kernel function and evaluate the output for the same x1
#  and x2 values. Notice that the result is the same in both scenarios
#  demonstrating the power of kernel trick.

import numpy as np
import math

def Transform(a, b):
    #Transforms two input values into a higher-dimensional space
    return np.array([
        a ** 2,
        math.sqrt(2) * a * b,
        b ** 2
    ])

def polynomial_kernel(a, b):
    #Kernel function that matches the transform function
    return (a[0]**2) * (b[0]**2) + 2 * a[0]*b[0]*a[1]*b[1] + (a[1]**2) * (b[1]**2)

def main():
    # Correcting input vectors (x1, x2)
    x1 = np.array([3, 10])
    x2 = np.array([6, 10])

    # Applying transformation
    phi_x1 = Transform(x1[0], x1[1])
    print("The transformed vec1 is:", phi_x1)

    phi_x2 = Transform(x2[0], x2[1])
    print("The transformed vec2 is:", phi_x2)

    # Dot product in transformed space
    dot_product = np.dot(phi_x1, phi_x2)
    print("Dot product in transformed space:", dot_product)

    # Applying kernel directly
    kernel_output = polynomial_kernel(x1, x2)
    print("Kernel function output:", kernel_output)

if __name__ == "__main__":
    main()