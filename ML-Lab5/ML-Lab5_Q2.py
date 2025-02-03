import matplotlib.pyplot as plt
import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def main():
    x=np.linspace(-10, 10)
    y=sigmoid_function(x)
    plt.plot(x, y, color="blue", label="sigmoid function")
    plt.show()

if __name__ == '__main__':
    main()