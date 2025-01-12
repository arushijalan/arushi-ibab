import matplotlib.pyplot as plt
import numpy as np
def graph(x, mean, sigma):
    b = (x-mean)/sigma
    c = sigma * np.sqrt(2 * np.pi)
    # y = (np.exp(-0.5*(b**2)))/c
    y = (1/c)*np.e**((-0.5)*b**2)
    fig, a = plt.subplots()
    a.plot(x, y)
    a.set_title("Gaussian PDF")
    a.set_xlabel("X - Axis")
    a.set_ylabel("Y - Axis")
    plt.show()
def main():
    g = np.linspace(-100, 100, 100)
    print(graph(g, 0, 15))
if __name__ == "__main__":
    main()