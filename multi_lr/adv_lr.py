import pickle

import matplotlib.pyplot as plt
from data import gen_data, shape_idx
import numpy as np
from lr import plot_data


def w_grad(R, x, y, w):
    m = x.shape[0]
    sigma = w * 0
    for x_i, y_i in zip(x, y):
        x_tilde = R.T @ x_i.reshape(-1, 1)
        sigma += ((1 / (1 + np.exp(-w.T @ x_tilde))) - y_i) * x_tilde
    return sigma / m


def R_grad(R, x, y, w):
    m = x.shape[0]
    sigma = R * 0
    for x_i, y_i in zip(x, y):
        x_tilde = R.T @ x_i
        sigma += ((1 / (1 + np.exp(-w.T @ x_tilde))) - y_i) * x_i.reshape(-1, 1) @ w.T
    return sigma / m


def find_w(R, x, y, lr=1e-1, tol=1e-2):
    r = R.shape[1]
    w = np.random.randn(r, 1)
    acc_list = []
    norm_list = []
    t = 0
    while True:
        dw = w_grad(R=R, x=x, y=y, w=w)
        dw_norm = np.linalg.norm(dw)
        norm_list.append(dw_norm)
        w = w - lr * dw
        acc = (((x @ R) @ w > 0) == (y.reshape(-1, 1) == 1)).mean()
        acc_list.append(acc)
        #print(dw_norm, acc)
        if dw_norm < tol or t > 1000:
            #print(dw_norm, acc)
            break
        t += 1
    return w


def find_R(w, x, y, r, T=100, lr=1e-1):
    d = x.shape[1]
    R = np.random.randn(d, r)
    acc_list = []
    #print("*****")
    for t in range(T):
        dR = R_grad(R=R, x=x, y=y, w=w)
        R = R - lr * dR
        if t % 10 == 0:
            acc = (((x @ R) @ w > 0) == (y.reshape(-1, 1) == 1)).mean()
            #print(acc)
    return R


def pca(X, num_components):
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :num_components]
    transformed_data = np.dot(X, top_eigenvectors) @ top_eigenvectors.T
    return np.real(transformed_data), np.real(top_eigenvectors), np.real(eigenvalues)


def pca_step(x, r, labels):
    pca_x, eigenvectors, eigenvalues = pca(X=x, num_components=r)
    # plt.imshow(transformed_data[0].reshape(25, 25))
    # plt.title(f"{n}")
    # plt.show()
    w = find_w(R=np.eye(625), x=pca_x, y=labels)
    acc = (((x @ np.eye(625)) @ w > 0) == (labels.reshape(-1, 1) == 1)).mean()
    return pca_x, w, acc


def algo_step(d, r, x, labels, R_dict, name, w_lr=1e-1, r_lr=1e0, tol=1e-2, T=100):
    R = np.random.randn(d, r)
    for i in range(4):
        w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
        R = find_R(w=w, x=x, y=labels, r=r, lr=r_lr, T=T)
    w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
    R_dict[name] = R, w
    acc = (((x @ R) @ w > 0) == (labels.reshape(-1, 1) == 1)).mean()
    print(f"Alg: {acc}")
    return R_dict


def solve_eq(R_dict, x, full_labels, w_lr=1e-1, tol=1e-2):
    a = len(shape_idx) - 1
    M = np.zeros([a, a])
    for R_idx in range(1, a + 1):
        name = shape_idx[R_idx]
        R, _w = R_dict[name]
        for f_idx in range(1, a + 1):
            labels = full_labels[:, f_idx]
            labels = (labels + 1) // 2
            w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
            acc = (((x @ R) @ w > 0) == (labels.reshape(-1, 1) == 1)).mean()
            M[R_idx - 1, f_idx - 1] = acc
    b = np.zeros(M.shape[0])
    b[-1] = 1
    A = np.zeros([a, a])
    for i in range(a - 1):
        A[i] = M[i] - M[i + 1]
    A[-1] = np.ones(a)
    x = np.linalg.inv(A) @ b
    eq = M @ x
    assert max(eq) - min(eq) < 1e-2
    print(eq[0])
    return eq[0]


def run_iter(N, r_list_pca, r_list_algo, num_shapes=6):
    data, full_labels = gen_data(N=N)
    x = data.reshape(data.shape[0], -1)
    d = x.shape[1]

    pca_acc = np.ones([len(r_list_pca), num_shapes])
    for r_idx, r in enumerate(r_list_pca):
        print(f"*** *** r={r}, pca")
        for f_idx, labels_idx in enumerate(range(1, num_shapes+1)):
            name = shape_idx[labels_idx]
            # plot_data(data=data, labels=labels, n=5, title=name)
            labels = full_labels[:, labels_idx]
            labels = (labels + 1) // 2

            pca_x, w, acc = pca_step(x=x, r=r, labels=labels)
            pca_acc[r_idx, f_idx] = acc

    algo_acc = np.ones([len(r_list_algo)])
    for r_idx, r in enumerate(r_list_algo):
        print(f"*** *** r={r}, algo")
        R_dict = {}
        for labels_idx in range(1, 7):
            name = shape_idx[labels_idx]
            labels = full_labels[:, labels_idx]
            labels = (labels + 1) // 2
            R_dict = algo_step(d, r, x, labels, R_dict=R_dict, name=name, w_lr=1e-1, r_lr=1e0, tol=1e-2, T=100)
        eq_val = solve_eq(R_dict=R_dict, x=x, full_labels=full_labels, w_lr=1e-1, tol=1e-2)
        algo_acc[r_idx] = eq_val

    return pca_acc, algo_acc


def plot(results):
    (r_list_pca, r_list_lago, pca_acc_mul, eq_val_mul) = results

    plt.plot(r_list_pca, pca_acc_mul.min(axis=2).mean(axis=0), label="Worst Case PCA Acc")
    plt.fill_between(r_list_pca, pca_acc_mul.min(axis=2).mean(axis=0) + pca_acc_mul.min(axis=2).std(axis=0),
                     pca_acc_mul.min(axis=2).mean(axis=0) - pca_acc_mul.min(axis=2).std(axis=0), alpha=0.2)

    plt.plot(r_list_pca, pca_acc_mul.mean(axis=2).mean(axis=0), label="Mean PCA Acc")
    plt.fill_between(r_list_pca, pca_acc_mul.mean(axis=2).mean(axis=0) + pca_acc_mul.mean(axis=2).std(axis=0),
                     pca_acc_mul.mean(axis=2).mean(axis=0) - pca_acc_mul.mean(axis=2).std(axis=0), alpha=0.2)

    for idx, algo_r in enumerate(r_list_lago):
        curr_eq_val = eq_val_mul[:, idx]
        plt.plot(r_list_pca, np.ones(len(r_list_pca)) * curr_eq_val.mean(), label=f"Algo Acc r={algo_r}")
        plt.fill_between(r_list_pca, np.ones(len(r_list_pca)) * (curr_eq_val.mean() + curr_eq_val.std()),
                         np.ones(len(r_list_pca)) * (curr_eq_val.mean() - curr_eq_val.std()), alpha=0.2)

    plt.ylabel("Accuracy")
    plt.xlabel("r")

    plt.grid()
    plt.legend()
    plt.savefig("results_acc_mul-loglosss.png")


if __name__ == "__main__":

    r_list_pca = range(1, 6)
    r_list_lago = range(1, 3)

    num_shapes = 6
    runs = 3
    N = 100

    pca_acc_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    eq_val_mul = np.zeros([runs, len(r_list_lago)])

    for i in range(runs):
        print(f"*** Run #{i}")
        pca_acc, eq_val = run_iter(N=N, num_shapes=6, r_list_pca=r_list_pca, r_list_algo=r_list_lago)
        eq_val_mul[i] = eq_val
        pca_acc_mul[i] = pca_acc

    with open("res_acc_mul_ll.p", "wb") as f:
        results = [r_list_pca, r_list_lago, pca_acc_mul, eq_val_mul]
        pickle.dump(results, f)

    plot(results=results)
