import matplotlib.pyplot as plt
import numpy as np
from Lab5_ex2 import sigmoid_function


def derivative_sigmoid(x):
    gz=sigmoid_function(x)
    return gz*(1-gz)


x_values = np.linspace(-10, 10)
y_values = sigmoid_function(x_values)
y=derivative_sigmoid(x_values)


plt.plot(x_values, y_values, label="Sigmoid Function", color="blue")
plt.plot(x_values, y, label="Sigmoid Derivative", color="red")
plt.title("Sigmoid Function and its derivative")
plt.xlabel("x")
plt.ylabel("Value")
plt.show()