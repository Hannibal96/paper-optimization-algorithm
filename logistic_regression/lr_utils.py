import numpy as np


def matrix_power(A, p):
    D, U = np.linalg.eig(A)
    return U @ (np.diag(D**p)) @ U.T


def regret(x, R, p, f, o, Q):
    regret = 0
    for j, R_j in enumerate(R):
        for i, f_i in enumerate(f):
            q_ji = Q[j, i]
            regret += p[j] * o[i] * loss(x=x, R=R_j, f=f_i, q=q_ji)
    return regret


def loss(x, R, f, q):
    p1 = calc_p1(f=f, x=x)
    p2 = calc_p2(x=x, R=R, q=q)
    return (p1 * np.log(p1 / p2) + (1-p1) * np.log((1-p1) / (1-p2))).mean(axis=0)


def find_q(x, R, f, lr, tol=1e-3):
    r = R.shape[1]
    q = np.random.randn(r, 1)

    for t in range(1_000_000):
        q_g = q_grad(x=x, q=q, R=R, f=f)
        q = q - lr * q_g
        if np.linalg.norm(q_g) < tol:
            break
        if (t+1) % 10_000 == 0:
            lr = lr/2
    return q


def q_grad(x, q, R, f):
    p1 = calc_p1(f=f, x=x)
    p2 = calc_p2(x=x, R=R, q=q)
    return ((p2 - p1) * R.T @ x).mean(axis=0)


def f_grad(x, R, f, p, Q):
    f_grad = f * 0
    for j, R_j in enumerate(R):
        f_grad += f_grad_aux(x=x, q=Q[j], R=R_j, f=f) * p[j]
    return f_grad


def f_grad_aux(x, q, R, f):
    f_tilde = R @ q
    p1 = calc_p1(f=f, x=x)
    return (p1 * (1-p1) * (f - f_tilde).T @ x * x).mean(axis=0)


def R_grad(x, Q, R, f, o):
    grad = R * 0
    for i, f_i in enumerate(f):
        q_ji = Q[i]
        grad += R_grad_aux(x=x, q=q_ji, R=R, f=f_i) * o[i]
    return grad


def R_grad_aux(x, q, R, f):
    p1 = calc_p1(f=f, x=x)
    p2 = calc_p2(x=x, R=R, q=q)
    return ((p2 - p1) * x @ q.T).mean(axis=0)


def calc_p1(f, x):
    return 1 / (1 + np.exp(-f.T @ x))


def calc_p2(x, R, q):
    return 1 / (1 + np.exp(-q.T @ R.T @ x))


def MWU(p1, losses, p2, beta, dim):
    if dim == 1:
        loss = losses @ p2
        loss[loss > 100] = 100
        loss[loss < -100] = -100
        p = p1 * beta ** loss

    if dim == 2:
        loss = (p1.T @ losses).T
        loss[loss > 100] = 100
        loss[loss < -100] = -100
        p = p2 * beta ** loss

    p = p / sum(p)
    return p


def calc_loss_matrix(x, R, Q, f):
    losses_matrix = np.zeros([len(R), len(f)])
    for j in range(len(R)):
        R_j = R[j]
        for i in range(len(f)):
            f_i = f[i]
            q_ji = Q[j, i]
            losses_matrix[j, i] = loss(x=x, R=R_j, f=f_i, q=q_ji)
    return losses_matrix


def plot_and_save(x_axis, mean_regrets, std_regrets,
          label_char, labels_val, x_label, y_label, title, name):
    plt.grid()
    for i in range(mean_regrets.shape[0]):
        plt.plot(x_axis, mean_regrets[i], label=label_char+f"={labels_val[i]}")
        plt.fill_between(x_axis, mean_regrets[i]-std_regrets[i], mean_regrets[i]+std_regrets[i],
                         color='r', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.clf()

    pickle.dump(mean_regrets, open(f'{name}_mean.p', 'wb'))
    pickle.dump(std_regrets, open(f'{name}_std.p', 'wb'))



