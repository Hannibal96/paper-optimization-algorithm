import numpy as np

M = np.array([
    [1.00, 0.40, 0.36],
    [0.40, 1.00, 0.52],
    [0.36, 0.52, 1.00]])

d = M.shape[0]
b = np.zeros(d)
b[-1] = 1

A = np.zeros([d, d])
for i in range(d-1):
    A[i] = M[i]-M[i+1]
A[-1] = np.ones(d)

print(A)

x = np.linalg.inv(A) @ b

print(x)

print(M @ x)