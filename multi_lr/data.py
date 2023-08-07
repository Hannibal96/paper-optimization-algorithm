import numpy as np
import matplotlib.pyplot as plt

shape_idx = {0: "circle", 1: "square", 2: "diamond", 3: "triangle", 4: "cross", 5: "x", 6: "shapes", }


def create_triangle(size):
    triangle = np.zeros([size, size])
    triangle[-1, :] = 1
    triangle[:, 0] = 1
    for i in range(size):
        for j in range(size):
            if i == j:
                triangle[i, j] = 1

    return triangle


def create_circle(size):
    radius = size // 2
    diameter = radius * 2
    circle = np.zeros((diameter, diameter))
    center = radius - 0.5
    for i in range(diameter):
        for j in range(diameter):
            if radius ** 2 - 8 <= (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                circle[i, j] = 1
    return circle


def create_square(size):
    square = np.ones((size, size))
    square[1:size-1, 1:size-1] = 0
    return square


def create_diamond(size):
    diamond = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            if center - 1 < abs(i - center) + abs(j - center) <= center:
                diamond[i, j] = 1
    return diamond


def create_cross(size):
    arr = np.zeros((size, size))
    center = size // 2
    arr[center, :] = 1
    arr[:, center] = 1
    return arr


def create_x(size):
    arr = np.zeros((size, size))
    arr[np.arange(size), np.arange(size)] = 1
    arr[np.arange(size), np.arange(size)[::-1]] = 1
    return arr


def gen_data(s=10, N=100, offset=False, noise=False):
    S = 25
    generators = [create_circle, create_square, create_diamond, create_triangle, create_cross, create_x]
    data = []
    labels = []
    for i in range(N):
        photo = np.zeros(shape=(S, S))

        num_shape = np.random.randint(low=1, high=5)
        Q = np.random.choice(range(4), num_shape, replace=False)
        flags = np.zeros(6, dtype=np.int32)
        for q in Q:
            g = np.random.randint(low=0, high=len(generators))
            shape = generators[g](size=s)
            flags[g] = 1
            x_offset = np.random.randint(low=-1, high=2) * int(offset)
            y_offset = np.random.randint(low=-1, high=2) * int(offset)
            if q == 0:
                photo[1+x_offset:1+s+x_offset, 1+y_offset:1+s+y_offset] = shape
            if q == 1:
                photo[1+x_offset:1+s+x_offset, 14+y_offset:14+s+y_offset] = shape
            if q == 2:
                photo[14+x_offset:14+s+x_offset, 1+y_offset:1+s+y_offset] = shape
            if q == 3:
                photo[14+x_offset:14+s+x_offset, 14+y_offset:14+s+y_offset] = shape
        photo = photo + int(noise) * np.random.randint(size=(S, S), low=0, high=100) / 200
        # num_shapes, 
        label = list(flags)
        label.append(num_shape)
        labels.append(label)
        data.append(photo)
    return np.array(data), np.array(labels)


def plot_data(data=None, labels=None, N=100, n=5):
    if data is None:
        data, labels = gen_data(N=N)

    for sample, (photo, curr_labels) in enumerate(zip(data[0:n], labels[0:n])):
        title = ""
        for labels_idx in range(len(shape_idx) - 1):
            shape_label = curr_labels[labels_idx]
            title += shape_idx[labels_idx] + f": {shape_label}, "
        plt.title(title)
        plt.imshow(photo)
        plt.savefig(f"./multi_lr/Lyx/sample_{sample}.png")
        plt.clf()


if __name__ == "__main__":
    data, labels = gen_data(s=10, N=100)
    x = data.reshape(data.shape[0], -1)
    y = labels[:, range(0, len(shape_idx)-1)]
    d = x.shape[1]
    r = 3
    m = x.shape[0]
    R = np.random.randn(d, r)
    w = np.random.randn(r, 1)


    plot_data(data=data, labels=labels)


