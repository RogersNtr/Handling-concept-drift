import matplotlib.pyplot as plt
import numpy as np


def plot_scatter_real(data1, data2, groups):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    colors = ("red", "green")
    groups = groups
    print(data1)
    data = (data1, data2)
    # for dat, color, group in zip(data, colors, groups):
    x = data1['current']
    y = data2['current']
    ax.scatter(x, y, alpha=0.8, c=colors, edgecolors='none', s=30, label=groups)

    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()


def plot_virtual_drift():

    # Create data
    N = 60
    g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
    g2 = (0.4 + 0.3 * np.random.rand(N), 0.5 * np.random.rand(N))
    g3 = (0.17 + 0.3 * np.random.rand(N), 0.3 * np.random.rand(N))
    print(g2[0].size)
    data = (g2, g3)
    colors = ("red", "green")  # , "blue"
    groups = ("feature 1", "feature 2")  #, "water"

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)

    plt.title('concept model distribution 2')
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend(loc=2)
    plt.show()


plot_virtual_drift()