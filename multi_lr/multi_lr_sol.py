import pickle
import matplotlib.pyplot as plt
from data import gen_data, shape_idx, plot_data
import numpy as np
import math
import argparse


def get_dw(R, x, y, w):
    m = x.shape[0]
    return ((1/(1+np.exp(-w.T @ R.T @ x.T)) - y) @ (R.T @ x.T).T) / m


def get_dR(R, x, y, w):
    m = x.shape[0]
    dr = (1 / (1 + np.exp(-w.T @ R.T @ x.T)) - y).reshape(m, 1, 1) * x.reshape(m, -1, 1) @ w.T
    return dr.mean(axis=0)


def find_w(R, x, y, lr=1e-1, tol=1e-2):
    r = R.shape[1]
    w = np.random.randn(r, 1)
    acc_list = []
    norm_list = []
    for t in range(1000):
        dw = get_dw(R=R, x=x, y=y, w=w).reshape(r, 1)
        w = w - lr * dw
        norm_list.append(np.linalg.norm(dw))
        loss = calc_loss(x=x, R=R, w=w, labels=y)
        acc_list.append(calc_acc(x=x, R=R, w=w, labels=y))
        if norm_list[-1] < tol:
            break
    return w


def find_R(w, x, y, r, T=100, lr=1e-1):
    d = x.shape[1]
    R = np.random.randn(d, r)
    for t in range(T):
        dR = get_dR(R=R, x=x, y=y, w=w)
        R = R - lr * dR
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
    w = find_w(R=np.eye(625), x=pca_x, y=labels)
    acc = calc_acc(x=x, R=np.eye(625), w=w, labels=labels)
    loss = calc_loss(x=x, R=np.eye(625), w=w, labels=labels)
    return pca_x, w, acc, loss


def algo_step(d, r, x, labels, R_dict, name, w_lr=1e-1, r_lr=1e0, tol=1e-2, T=100):
    R = np.random.randn(d, r)
    for i in range(4):
        w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
        R = find_R(w=w, x=x, y=labels, r=r, lr=r_lr, T=T)
    w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
    R_dict[name] = R, w
    acc = calc_acc(x=x, R=R, w=w, labels=labels)
    loss = calc_loss(x=x, R=R, w=w, labels=labels)
    print(f"Alg: {acc}")
    return R_dict


def solve_eq(R_dict, x, full_labels, w_lr=1e-1, tol=1e-2):
    a = len(shape_idx) - 1
    M_acc = np.zeros([a, a])
    M_loss = np.zeros([a, a])
    for R_idx in range(a):
        name = shape_idx[R_idx]
        R, _ = R_dict[name]
        for f_idx in range(1, a + 1):
            labels = full_labels[:, f_idx]
            labels = (labels + 1) // 2
            w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
            #acc = (((x @ R) @ w > 0) == (labels.reshape(-1, 1) == 1)).mean()
            acc = calc_acc(x=x, R=R, w=w, labels=labels)
            loss = calc_loss(x=x, R=R, w=w, labels=labels)
            M_acc[R_idx - 1, f_idx - 1] = acc
            M_loss[R_idx - 1, f_idx - 1] = loss

    b = np.zeros(a)
    b[-1] = 1
    A_acc = np.zeros([a, a])
    A_loss = np.zeros([a, a])
    for i in range(a - 1):
        A_acc[i] = M_acc[i] - M_acc[i + 1]
        A_loss[i] = M_loss[i] - M_loss[i + 1]
    A_acc[-1] = np.ones(a)
    A_loss[-1] = np.ones(a)

    x_acc = np.linalg.inv(A_acc) @ b
    x_loss = np.linalg.inv(A_loss) @ b

    eq_acc = M_acc @ x_acc
    eq_loss = M_loss @ x_loss

    assert max(eq_acc) - min(eq_acc) < 1e-2
    assert max(eq_loss) - min(eq_loss) < 1e-2

    return eq_acc[0], eq_loss[0]


def calc_acc(x, R, w, labels):
    acc = (((x @ R) @ w > 0) == (labels.reshape(-1, 1) == 1)).mean()
    return acc


def calc_loss(x, R, w, labels):
    col_y = labels.reshape(-1, 1)
    q = 1/(1+np.exp(-(x @ R) @ w))
    l1 = np.log(q)
    l1[l1 < -100] = -100
    l2 = np.log(1 - q)
    l2[l2 < -100] = -100
    loss = -(col_y * l1 + (1 - col_y) * l2)
    if math.isnan(loss.mean()):
        print("")
    return loss.mean()


def run_iter(N, r_list_pca, r_list_algo, num_shapes=6):
    data, full_labels = gen_data(N=N)
    x = data.reshape(data.shape[0], -1)
    d = x.shape[1]

    pca_acc = np.ones([len(r_list_pca), num_shapes])
    pca_loss = np.ones([len(r_list_pca), num_shapes])
    for r_idx, r in enumerate(r_list_pca):
        print(f"*** *** r={r}, pca")
        for f_idx, labels_idx in enumerate(range(len(shape_idx)-1)):
            name = shape_idx[labels_idx]
            # plot_data(data=data, labels=labels, n=5, title=name)
            labels = full_labels[:, labels_idx]
            pca_x, w, acc, loss = pca_step(x=x, r=r, labels=labels)
            pca_acc[r_idx, f_idx] = acc
            pca_loss[r_idx, f_idx] = loss

    algo_acc = np.ones([len(r_list_algo)])
    algo_loss = np.ones([len(r_list_algo)])
    for r_idx, r in enumerate(r_list_algo):
        print(f"*** *** r={r}, algo")
        R_dict = {}
        for labels_idx in range(len(shape_idx)-1):
            name = shape_idx[labels_idx]
            labels = full_labels[:, labels_idx]
            R_dict = algo_step(d, r, x, labels, R_dict=R_dict, name=name, w_lr=1e-1, r_lr=1e0, tol=1e-2, T=100)
        eq_acc, eq_loss = solve_eq(R_dict=R_dict, x=x, full_labels=full_labels, w_lr=1e-1, tol=1e-2)
        algo_acc[r_idx] = eq_acc
        algo_loss[r_idx] = eq_loss

    return pca_acc, pca_loss, algo_acc, algo_loss


def plot_results(results):
    (r_list_pca, r_list_lago, pca_acc_mul, pca_loss_mul, eq_acc_mul, eq_loss_mul) = results

    avg_res = pca_acc_mul.mean(axis=0)
    std_res = pca_acc_mul.std(axis=0)

    ind_res = np.argmin(avg_res, axis=1)
    min_res = avg_res[range(avg_res.shape[0]), ind_res]
    min_std = std_res[range(avg_res.shape[0]), ind_res]
    plt.plot(r_list_pca, min_res, label="Worst Case PCA Acc")
    plt.fill_between(r_list_pca, min_res + min_std, min_res - min_std, alpha=0.2)

    avg_mean = pca_acc_mul.mean(axis=0).mean(axis=1)
    std_mean = pca_acc_mul.std(axis=0).mean(axis=1)
    plt.plot(r_list_pca, avg_mean, label="Mean PCA Acc")
    plt.fill_between(r_list_pca, avg_mean + std_mean, avg_mean - std_mean, alpha=0.2)

    for idx, algo_r in enumerate(r_list_lago):
        curr_eq_val = eq_acc_mul[:, idx]
        plt.plot(r_list_pca, np.ones(len(r_list_pca)) * curr_eq_val.mean(), label=f"Algo Acc r={algo_r}")
        plt.fill_between(r_list_pca, np.ones(len(r_list_pca)) * (curr_eq_val.mean() + curr_eq_val.std()),
                         np.ones(len(r_list_pca)) * (curr_eq_val.mean() - curr_eq_val.std()), alpha=0.2)

    plt.ylabel("Accuracy")
    plt.xlabel("r")

    plt.grid()
    plt.legend()
    plt.savefig("results_acc_mul-lr.png")
    plt.clf()

    avg_res = pca_loss_mul.mean(axis=0)
    std_res = pca_loss_mul.std(axis=0)

    ind_res = np.argmax(avg_res, axis=1)
    min_res = avg_res[range(avg_res.shape[0]), ind_res]
    min_std = std_res[range(avg_res.shape[0]), ind_res]
    plt.plot(r_list_pca, min_res, label="Worst Case PCA Loss")
    plt.fill_between(r_list_pca, min_res + min_std, min_res - min_std, alpha=0.2)

    avg_mean = pca_loss_mul.mean(axis=0).mean(axis=1)
    std_mean = pca_loss_mul.std(axis=0).mean(axis=1)
    plt.plot(r_list_pca, avg_mean, label="Mean PCA Loss")
    plt.fill_between(r_list_pca, avg_mean + std_mean, avg_mean - std_mean, alpha=0.2)

    for idx, algo_r in enumerate(r_list_lago):
        curr_eq_val = eq_loss_mul[:, idx]
        plt.plot(r_list_pca, np.ones(len(r_list_pca)) * curr_eq_val.mean(), label=f"Algo Acc r={algo_r}")
        plt.fill_between(r_list_pca, np.ones(len(r_list_pca)) * (curr_eq_val.mean() + curr_eq_val.std()),
                         np.ones(len(r_list_pca)) * (curr_eq_val.mean() - curr_eq_val.std()), alpha=0.2)

    plt.ylabel("Loss")
    plt.xlabel("r")

    plt.grid()
    plt.legend()
    plt.savefig("results_loss_mul-lr.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=10)
    parser.add_argument("--N", default=10)

    args = parser.parse_args()

    num_shapes = len(shape_idx) - 1
    r_list_pca = range(1, 22, 2)
    r_list_lago = range(1, 7, 2)
    runs = args.runs
    N = args.N

    pca_acc_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    pca_loss_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    eq_acc_mul = np.zeros([runs, len(r_list_lago)])
    eq_loss_mul = np.zeros([runs, len(r_list_lago)])

    for i in range(runs):
        print(f"*** Run #{i}")
        pca_acc, pca_loss, algo_acc, algo_loss = run_iter(N=N, num_shapes=6, r_list_pca=r_list_pca, r_list_algo=r_list_lago)
        eq_acc_mul[i] = algo_acc
        eq_loss_mul[i] = algo_loss
        pca_acc_mul[i] = pca_acc
        pca_loss_mul[i] = pca_loss

    with open("res_acc_mul_ll.p", "wb") as f:
        results = [r_list_pca, r_list_lago, pca_acc_mul, pca_loss_mul, eq_acc_mul, eq_loss_mul]
        pickle.dump(results, f)

    plot_results(results=results)
