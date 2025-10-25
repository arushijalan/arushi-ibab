# 1. Implement the following functions in Python from scratch. Do not use any library functions. 
# You are allowed to use numpy and matplotlib. Generate 100 equally spaced values between -10 and 10. 
# Call this list as  z. Implement the following functions and its derivative. 
# Use class notes to find the expression for these functions. 
# Use z as input and plot both the function outputs and its derivative outputs.  
# Upload your code into Github and share it with me. 
# a. Sigmoid 
# b. Tanh 
# c. ReLU (Rectified Linear Unit) 
# d. Leaky ReLU 
# e. Softmax (no need for visualization) 

# 2. Write down the observations from the plot for all the above functions in the code. 
# a. What are the min and max values for the functions? 
# b. Is the output of the function zero-centred? 
# c. What happens to the gradient when the input values are too small or too big? 
# d. What is the relationship between sigmoid and tanh?

import numpy as np
import matplotlib.pyplot as plt

# Activation Functions and Derivatives

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of Sigmoid function
def deri_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

# Tanh function
def tanh(z):
    return np.tanh(z)

# Derivative of Tanh function
def deri_tanh(z):
    return 1 - np.tanh(z)**2

# ReLU function
def ReLU(z):
    return np.maximum(0, z)

# Derivative of ReLU function
def deri_ReLU(z):
    return np.where(z > 0, 1, 0)

# Leaky ReLU function
def leaky_ReLU(z, alpha=0.1):
    return np.where(z > 0, z, alpha * z)

# Derivative of Leaky ReLU function
def deri_leaky_ReLU(z, alpha=0.1):
    return np.where(z > 0, 1, alpha)

# Softmax function 
def softmax(z):
    exp_z = np.exp(z - np.max(z))  
    return exp_z / np.sum(exp_z)

def main():
    z = np.linspace(-10, 10, 100) # Generates 100 equally spaced values between -10 and 10

    # Sigmoid 
    plt.plot(z, sigmoid(z), label="Sigmoid")
    plt.plot(z, deri_sigmoid(z), label="Derivative")
    plt.title("Sigmoid Function and Derivative")
    plt.legend()
    plt.grid()
    plt.show()

    # Tanh 
    plt.plot(z, tanh(z), label="Tanh")
    plt.plot(z, deri_tanh(z), label="Derivative")
    plt.title("Tanh Function and Derivative")
    plt.legend()
    plt.grid()
    plt.show()

    # ReLU 
    plt.plot(z, ReLU(z), label="ReLU")
    plt.plot(z, deri_ReLU(z), label="Derivative")
    plt.title("ReLU Function and Derivative")
    plt.legend()
    plt.grid()
    plt.show()

    # Leaky ReLU
    plt.plot(z, leaky_ReLU(z), label="Leaky ReLU")
    plt.plot(z, deri_leaky_ReLU(z), label="Derivative")
    plt.title("Leaky ReLU Function and Derivative")
    plt.legend()
    plt.grid()
    plt.show()

    # Softmax (not plotted, just printed)
    example_input = np.array([1.0, 2.0, 3.0])
    print("Softmax of", example_input, "=", softmax(example_input))

# OBSERVATIONS
"""
Observations from the plots:

1. Sigmoid:
   - Range: (0, 1)
   - Not zero-centered (always positive)
   - Gradient vanishes for very large +ve or -ve inputs (close to 0)

2. Tanh:
   - Range: (-1, 1)
   - Zero-centered (better than sigmoid for optimization)
   - Gradient vanishes for very large +ve or -ve inputs
   - Relation with Sigmoid: tanh(x) = 2*sigmoid(2x) - 1

3. ReLU:
   - Range: [0, infinite)
   - Not zero-centered
   - Gradient is 1 for positive inputs, 0 for negative inputs
   - Can cause "dead neurons" (when many values < 0)

4. Leaky ReLU:
   - Range: (-inf, inf)
   - Almost zero-centered (small negative slope helps)
   - Gradient is aplha (for eg, 0.1) for negative inputs, 1 for positive inputs
   - Solves "dying ReLU" problem.

5. Softmax:
   - Outputs a probability distribution (values between 0 and 1 that sum to 1)
"""

if __name__ == "__main__":
    main()
