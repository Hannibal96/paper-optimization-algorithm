import pickle

import numpy as np

from multi_lr_sol import *
from mse.utils import MWU


def calc_loss_matrix(x, R, Q, f):
    losses_matrix = np.zeros([len(R), len(f)])
    for j, R_j in enumerate(R):
        for i, f_i in enumerate(f):
            q_ji = Q[j, i]
            losses_matrix[j, i] = calc_loss(x=x, R=R_j, w=q_ji, labels=f_i)
    return losses_matrix


def partial_grad_R(x, R_j, f, o, Q_j):
    dr = R_j * 0
    for f_i, q_ji, o_i in zip(f, Q_j, o):
        dr += get_dR(R=R_j, x=x, y=f_i, w=q_ji) * o_i
    return dr


def find_new_f(full_labels, data, R, p):
    worst = 0
    worst_f = None

    for f_idx, labels in enumerate(full_labels.T[0:-1]):
        total_loss = 0
        for curr_R, curr_p in zip(R, p):
            w = find_w(R=curr_R, x=data, y=labels)
            loss = calc_loss(x=x, R=R, w=w, labels=labels) * curr_p
            total_loss += curr_p * loss
        if total_loss > worst:
            worst = total_loss
            worst_f = f_idx
    return full_labels[:, worst_f]


def find_new_R(x, R, f, T, beta_f, beta_r, avg_frac, stop_frac, lr):

    T_stop = T - T // stop_frac
    T_avg = T // avg_frac
    d = R[0].shape[0]
    r = R[0].shape[1]

    o = np.ones([len(f), 1]) / len(f)
    p = np.ones([len(R) + 1, 1]) / (len(R) + 1)
    o_sum = o * 0
    p_sum = p * 0

    new_R = np.random.randn(d, r)
    R.append(new_R)

    W = np.zeros([len(R), len(f), r, 1])
    for i, f_i in enumerate(f):
        for j, R_j in enumerate(R):
            W[j, i] = find_w(R=R_j, x=x, y=f_i)

    for t in range(T):
        new_R = new_R - lr * partial_grad_R(x=x, R_j=new_R, f=f, o=o, Q_j=W[-1, :])

        R[-1] = new_R

        for i, f_i in enumerate(f):
            for j, R_j in enumerate(R):
                W[j, i] = find_w(R=R_j, x=x, y=f_i)

        L = calc_loss_matrix(x=x, R=R, Q=W, f=f)

        if t < T_stop:
            o = MWU(p1=p, losses=-L, p2=o, beta=beta_f, dim=2)
            p = MWU(p1=p, losses=L, p2=o, beta=beta_r, dim=1)
            if t >= T_stop - T_avg:
                o_sum = o_sum + o
                p_sum = p_sum + p
        if t == T_stop:
            try:
                o = o_sum / sum(o_sum)
                p = p_sum / sum(p_sum)
            except:
                o = np.ones([len(f), 1]) / len(f)
                p = np.ones([len(R), 1]) / len(R)

    return new_R, p, o

def mu_lr_algo(data, full_labels):
    R = np.random.randn(d, 2) * 0
    R = [R]
    f = []
    for i in range(5):
        print(i)
        new_f = find_new_f(full_labels=full_labels, data=data, R=R, p=[1])
        f.append(new_f)
        new_R, p, o = find_new_R(x=data, R=R, f=f, T=10, beta_f=0.95, beta_r=0.95, avg_frac=4, stop_frac=4, lr=1e-3)
        R.append(new_R)

    return R, f, p, o


if __name__ == "__main__":
    data, full_labels = gen_data(N=100, offset=True, noise=True)
    x = data.reshape(data.shape[0], -1)
    d = x.shape[1]
    R, f, p, o = mu_lr_algo(data=x, full_labels=full_labels)
    results = (R, f, p, o)
    with open("algo_res.p", "wb") as file:
        pickle.dump(results, file)



