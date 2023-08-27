import pickle
import matplotlib.pyplot as plt
from data import gen_data, shape_idx, plot_data, gen_mnist_data
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


def pca(X, test_x, num_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :num_components]
    transformed_data = np.dot(X, top_eigenvectors) @ top_eigenvectors.T
    transformed_test = np.dot(test_x, top_eigenvectors) @ top_eigenvectors.T
    return np.real(transformed_data), np.real(transformed_test)


def pca_step(x, test_x, r, labels, test_labels):
    pca_x, pca_test = pca(X=x, test_x=test_x, num_components=r)

    w = find_w(R=np.eye(x.shape[1]), x=pca_x, y=labels)
    acc = calc_acc(x=x, R=np.eye(x.shape[1]), w=w, labels=labels)
    loss = calc_loss(x=x, R=np.eye(x.shape[1]), w=w, labels=labels)

    test_w = find_w(R=np.eye(test_x.shape[1]), x=pca_test, y=test_labels)
    test_acc = calc_acc(x=test_x, R=np.eye(x.shape[1]), w=test_w, labels=test_labels)
    test_loss = calc_loss(x=test_x, R=np.eye(x.shape[1]), w=test_w, labels=test_labels)

    return pca_x, pca_test, w, test_w, acc, loss, test_acc, test_loss


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


def _build_m(a, x, R_dict, full_labels, w_lr, tol):
    M_acc = np.zeros([a, a])
    M_loss = np.zeros([a, a])
    for R_idx in range(a):
        name = shape_idx[R_idx]
        R, _ = R_dict[name]
        for f_idx in range(0, a):
            labels = full_labels[:, f_idx]
            w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
            acc = calc_acc(x=x, R=R, w=w, labels=labels)
            loss = calc_loss(x=x, R=R, w=w, labels=labels)
            M_acc[R_idx - 1, f_idx - 1] = acc
            M_loss[R_idx - 1, f_idx - 1] = loss
    return M_acc, M_loss


def _sol_eq(a, M_acc, M_loss):
    b = np.zeros(a)
    b[-1] = 1
    A_acc = np.zeros([a, a])
    A_loss = np.zeros([a, a])
    for i in range(a - 1):
        A_acc[i] = M_acc[i] - M_acc[i + 1]
        A_loss[i] = M_loss[i] - M_loss[i + 1]
    A_acc[-1] = np.ones(a)
    A_loss[-1] = np.ones(a)

    p_acc = np.linalg.inv(A_acc) @ b
    p_loss = np.linalg.inv(A_loss) @ b

    return p_acc, p_loss


def solve_eq(R_dict, x, x_test, full_labels, full_labels_test, w_lr=1e-1, tol=1e-2):

    a = full_labels.shape[1]
    M_acc, M_loss = _build_m(a=a, x=x, R_dict=R_dict, full_labels=full_labels, w_lr=w_lr, tol=tol)
    M_acc_test, M_loss_test = _build_m(a=a, x=x_test, R_dict=R_dict, full_labels=full_labels_test, w_lr=w_lr, tol=tol)

    p_acc, p_loss = _sol_eq(a=a, M_acc=M_acc, M_loss=M_loss)

    eq_acc = M_acc @ p_acc
    eq_loss = M_loss @ p_loss

    eq_acc_test = M_acc_test @ p_acc
    eq_loss_test = M_loss_test @ p_loss

    assert max(eq_acc) - min(eq_acc) < 1e-2
    assert max(eq_loss) - min(eq_loss) < 1e-2

    return eq_acc[0], eq_loss[0], min(eq_acc_test), max(eq_loss_test)


def calc_acc(x, R, w, labels):
    acc = (((x @ R) @ w > 0) == (labels.reshape(-1, 1) == 1)).mean()
    return acc


def calc_loss(x, R, w, labels):
    assert min(labels) == 0
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


def run_iter(r_list_pca, r_list_algo, num_shapes, data, full_labels,
             test_data, test_full_labels):

    x = data.reshape(data.shape[0], -1)
    x_test = test_data.reshape(test_data.shape[0], -1)
    d = x.shape[1]

    pca_acc = np.ones([len(r_list_pca), num_shapes])
    pca_loss = np.ones([len(r_list_pca), num_shapes])
    pca_test_acc = np.ones([len(r_list_pca), num_shapes])
    pca_test_loss = np.ones([len(r_list_pca), num_shapes])
    for r_idx, r in enumerate(r_list_pca):
        print(f"*** *** r={r}, pca")
        for labels_idx in range(full_labels.shape[1]):
            labels = full_labels[:, labels_idx]
            test_labels = test_full_labels[:, labels_idx]
            pca_x, pca_test, w, test_w, acc, loss, test_acc, test_loss = pca_step(x=x, test_x=x_test, r=r, labels=labels, test_labels=test_labels)

            pca_acc[r_idx, labels_idx] = acc
            pca_loss[r_idx, labels_idx] = loss
            pca_test_acc[r_idx, labels_idx] = test_acc
            pca_test_loss[r_idx, labels_idx] = test_loss

    algo_acc = np.ones([len(r_list_algo)])
    algo_loss = np.ones([len(r_list_algo)])
    algo_test_acc = np.ones([len(r_list_algo)])
    algo_test_loss = np.ones([len(r_list_algo)])
    for r_idx, r in enumerate(r_list_algo):
        print(f"*** *** r={r}, algo")
        R_dict = {}
        for labels_idx in range(full_labels.shape[1]):
            name = shape_idx[labels_idx]
            labels = full_labels[:, labels_idx]
            R_dict = algo_step(d, r, x, labels, R_dict=R_dict, name=name, w_lr=1e-1, r_lr=1e0, tol=1e-2, T=100)

        eq_acc, eq_loss, eq_acc_test, eq_loss_test = solve_eq(R_dict=R_dict, x=x, x_test=x_test,
                                                                              full_labels=full_labels, full_labels_test=test_full_labels, w_lr=1e-1, tol=1e-2)
        algo_acc[r_idx] = eq_acc
        algo_loss[r_idx] = eq_loss
        algo_test_acc[r_idx] = eq_acc_test
        algo_test_loss[r_idx] = eq_loss_test

    return pca_acc, pca_loss, pca_test_acc, pca_test_loss, algo_acc, algo_loss, algo_test_acc, algo_test_loss


def plot_results(results, path):
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
    plt.savefig(f"./graphs/Acc_{path}.png")
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
    plt.savefig(f"./graphs/Loss_{path}.png")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=5, type=int, required=True)
    parser.add_argument("--N", default=100, type=int, required=True)
    parser.add_argument("--data", "-d", choices=['mnist', 'syn'], required=True)
    parser.add_argument("--offset", "-o", required=False, action='store_true', default=False)
    parser.add_argument("--noise", "-n", required=False, action='store_true', default=False)
    parser.add_argument("--not_train", required=False, action='store_true', default=False)

    args = parser.parse_args()

    if args.data == "syn":
        num_shapes = len(shape_idx) - 1
    else:
        num_shapes = 10

    r_list_pca = range(1, 22, 2)
    r_list_lago = range(1, 7, 2)
    runs = args.runs
    N = args.N

    pca_acc_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    pca_loss_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    eq_acc_mul = np.zeros([runs, len(r_list_lago)])
    eq_loss_mul = np.zeros([runs, len(r_list_lago)])
    pca_acc_test_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    pca_loss_test_mul = np.zeros([runs, len(r_list_pca), num_shapes])
    eq_acc_test_mul = np.zeros([runs, len(r_list_lago)])
    eq_loss_test_mul = np.zeros([runs, len(r_list_lago)])

    for i in range(runs):
        print(f"*** Run #{i}")
        if args.data == "syn":
            train_data, train_full_labels = gen_data(N=N, noise=args.noise, offset=args.offset)
            test_data, test_full_labels = gen_data(N=N//2, noise=args.noise, offset=args.offset)
        else:
            train_data, train_full_labels = gen_mnist_data(N=N)
            test_data, test_full_labels = gen_mnist_data(N=N)
            shape_idx = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
        pca_acc, pca_loss, pca_test_acc, pca_test_loss, algo_acc, algo_loss, algo_test_acc, algo_test_loss = \
            run_iter(num_shapes=num_shapes, r_list_pca=r_list_pca, r_list_algo=r_list_lago, data=train_data, full_labels=train_full_labels, test_data=test_data, test_full_labels=test_full_labels)

        eq_acc_mul[i] = algo_acc
        eq_loss_mul[i] = algo_loss
        pca_acc_mul[i] = pca_acc
        pca_loss_mul[i] = pca_loss

        eq_acc_test_mul[i] = algo_test_acc
        eq_loss_test_mul[i] = algo_test_loss
        pca_acc_test_mul[i] = pca_test_acc
        pca_loss_test_mul[i] = pca_test_loss

    with open(f"./pickles/Results_D={args.data}_N={N}_R={runs}.p", "wb") as f:
        results = [r_list_pca, r_list_lago, pca_acc_mul, pca_loss_mul, eq_acc_mul, eq_loss_mul, pca_acc_test_mul, pca_loss_test_mul, eq_acc_test_mul, eq_loss_test_mul]
        pickle.dump(results, f)

    r1 = [r_list_pca, r_list_lago, pca_acc_mul, pca_loss_mul, eq_acc_mul, eq_loss_mul]
    r2 = [r_list_pca, r_list_lago, pca_acc_test_mul, pca_loss_test_mul, eq_acc_test_mul, eq_loss_test_mul]

    path = f"D={args.data}{int(args.offset) * '_offset'}{int(args.noise) * '_noise'}_N={N}_R={runs}"
    plot_results(results=r1, path=f"train_{path}")
    plot_results(results=r2, path=f"test_D={path}")
