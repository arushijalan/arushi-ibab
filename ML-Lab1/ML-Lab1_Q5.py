import matplotlib.pyplot as plt
import numpy as np
# import sympy as sp
def graph(x):
    y = x**2
    y1 = 2*x
    fig, a = plt.subplots()
    a.plot(x, y)
    a.plot(x, y1)
    a.set_title("Implement x**2 and its derivative")
    a.set_xlabel("X - Axis")
    a.set_ylabel("Y - Axis")
    plt.show()
def main():
    g = np.linspace(-100, 100, 100)
    print(graph(g))
if __name__ == "__main__":
    main()