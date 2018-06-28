import numpy as np
import matplotlib.pyplot as plt

def binary_step(x):
    return np.where(x >= 0.0, 1, 0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def plot_activation(x, y, yrange=(0, 1)):
    plt.plot(x, y)
    plt.axvline(0.0, color='k')
    plt.ylim(yrange[0]-0.1, yrange[1]+0.1)
    plt.yticks(np.arange(yrange[0], yrange[1]+0.1, 0.5))
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

def plot_binary_step():
    x = np.arange(-2, 2, 0.001)
    y = binary_step(x)
    plot_activation(x, y)

def plot_sigmoid():
    x = np.arange(-7, 7, 0.1)
    y = sigmoid(x)
    plot_activation(x, y)

def plot_tanh():
    x = np.arange(-7, 7, 0.1)
    y = tanh(x)
    plot_activation(x, y, yrange=(-1, 1))

def plot_relu():
    x = np.arange(-2, 2, 0.1)
    y = relu(x)
    plot_activation(x, y)
