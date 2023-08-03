import numpy as np
import matplotlib.pyplot as plt

shape_idx = {0: "shapes", 1: "circle", 2: "square", 3: "diamond", 4: "triangle", 5: "cross", 6: "x"}


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


def gen_data(s=10, N=100):
    S = 25
    generators = [create_circle, create_square, create_diamond, create_triangle, create_cross, create_x]
    data = []
    labels = []
    for i in range(N):
        photo = np.zeros(shape=(S, S))

        num_shape = np.random.randint(low=1, high=5)
        Q = np.random.choice(range(4), num_shape, replace=False)
        flags = -np.ones(6, dtype=np.int32)
        for q in Q:
            g = np.random.randint(low=0, high=len(generators))
            shape = generators[g](size=s)
            flags[g] = 1

            if q == 0:        photo[1:1+s, 1:1+s] = shape
            if q == 1:        photo[1:1+s, 14:14+s] = shape
            if q == 2:        photo[14:14+s, 1:1+s] = shape
            if q == 3:        photo[14:14+s, 14:14+s] = shape

        # num_shapes, 
        label = list(flags)
        label.insert(0, num_shape)
        labels.append(label)
        data.append(photo)
    return np.array(data), np.array(labels)

