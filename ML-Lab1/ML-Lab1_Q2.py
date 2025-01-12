import matplotlib.pyplot as plt
import numpy as np
def graph(x):
    y = 2*x + 3
    fig, a = plt.subplots()
    a.plot(x, y)
    a.set_title("Implement y = 2x + 3")
    a.set_xlabel("X - Axis")
    a.set_ylabel("Y - Axis")
    plt.show()
def main():
    g = np.linspace(-100, 100, 100)
    print(graph(g))
if __name__ == "__main__":
    main()  