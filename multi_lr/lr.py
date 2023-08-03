import matplotlib.pyplot as plt
from data import gen_data, shape_idx
import numpy as np


def w_grad(x, y, w):
    m = x.shape[0]
    sigma = 0
    for x_i, y_i in zip(x, y):
        sigma += y_i * x_i / (1+np.exp(-np.dot(x_i, w) * y_i))
    return -sigma / m


def lr(data, labels, lr=1e-1, tol=1e-2):
    x = data.reshape(data.shape[0], -1)
    w = np.random.randn(x.shape[1])
    acc_list = []
    norm_list = []
    while True:
        dw = w_grad(x=x, y=labels, w=w)
        if np.linalg.norm(dw) < tol:
            break
        w = w + lr * dw
        acc = ((np.dot(x, w) > 0) == (labels == -1)).mean()
        w_norm = np.linalg.norm(dw)
        if len(acc_list) % 100 == 0:
            print(f"||w|| = {w_norm}")
            print(f"acc: {acc * 100}%")
        acc_list.append(acc * 100)
        norm_list.append(w_norm)
    plt.plot(acc_list)
    plt.show()
    plt.plot(norm_list)
    plt.show()
    return w


def plot_data(data, labels, n, title):
    for photo, label in zip(data[0:n], labels[0:n]):
        plt.title(title+": "+str(label))
        plt.imshow(photo)
        plt.show()


if __name__ == "__main__":

    data, labels = gen_data(N=100)
    labels_idx = 6
    labels = labels[:, labels_idx]
    #plot_data(data=data, labels=labels, n=10, title=shape_idx[labels_idx])

    w = lr(data=data, labels=labels, tol=1e-1)

    data, labels = gen_data(N=1000)
    labels = labels[:, 2]
    x = data.reshape(data.shape[0], -1)
    acc = ((np.dot(x, w) > 0) == (labels == -1)).mean()
    print(acc * 100)

    """
    
    """



