import matplotlib.pyplot as plt
import numpy as np
def graph(x):
    y = 2*x**2 + 3*x + 4
    fig, a = plt.subplots()
    a.plot(x, y)
    a.set_title("Implement y = 2x^2 + 3x + 4")
    a.set_xlabel("X - Axis")
    a.set_ylabel("Y - Axis")
    plt.show()
def main():
    g = np.linspace(-10, 10, 100)
    print(graph(g))
if __name__ == "__main__":
    main()