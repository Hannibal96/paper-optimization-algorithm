import numpy as np
from tqdm import tqdm
from lr_utils import *


def find_Q(x, R, f, lr, tol=1e-3):
    r = R[0].shape[1]
    Q = np.zeros([len(R), len(f), r, 1])
    for j, R_j in enumerate(R):
        for i, f_i in enumerate(f):
            q_ji = find_q(x=x, R=R_j, f=f_i, lr=lr, tol=tol)
            Q[j, i] = q_ji.copy()
    return Q


def find_f(x, R, p, lr, T=1000, tol=1e-3):
    d = x.shape[1]
    f = np.random.randn(d, 1)
    f_prev = f.copy()
    #for t in tqdm(range(T)):
    for t in range(T):
        Q = find_Q(x=x, R=R, f=[f], lr=lr, tol=tol)
        f_g = f_grad(x=x, R=R, f=f, p=p, Q=Q[:, 0])
        f = f + lr * f_g
        if np.linalg.norm(f - f_prev) < tol:
            break
        if np.linalg.norm(f) > 1:
            f = f / np.linalg.norm(f)
        f_prev = f.copy()
    reg = regret(x=x, R=R, p=p, f=[f], o=np.ones(1), Q=Q)
    return f, reg


def find_R(x, R, f, T, lr_r, lr_f, beta1, beta2, avg_frac, stop_frac, tol=1e-3):
    R_pool = R[:]
    T_stop = T - T // stop_frac
    T_avg = T // avg_frac
    d = R[0].shape[0]
    r = R[0].shape[1]

    o = np.ones([len(f), 1]) / len(f)
    p = np.ones([len(R_pool)+1, 1]) / (len(R_pool)+1)

    o_sum = o * 0
    p_sum = p * 0

    new_R = np.random.randn(d, r)
    R_pool.append(new_R)
    Q = find_Q(x=x, R=R_pool, f=f, lr=lr_f, tol=tol)

    #for t in tqdm(range(T)):
    for t in range(T):
        new_R = new_R - lr_r * R_grad(x=x, Q=Q[-1, :], R=new_R, f=f, o=o)

        R_pool[-1] = new_R
        Q[-1, :] = find_Q(x=x, R=[new_R], f=f, lr=lr_f, tol=tol)
        L = calc_loss_matrix(x=x, R=R_pool, Q=Q, f=f)

        if t < T_stop:
            o = MWU(p1=p, losses=-L, p2=o, beta=beta1, dim=2)
            p = MWU(p1=p, losses=L, p2=o, beta=beta2, dim=1)
            if t >= T_stop - T_avg:
                o_sum = o_sum + o
                p_sum = p_sum + p
        if t == T_stop:
            o = o_sum / sum(o_sum)
            p = p_sum / sum(p_sum)

    return new_R, p, o


def init_R0_f0(x, r, lr_f, T_f, tol=1e-3):
    d = x.shape[1]
    R0 = np.random.randn(d, r) * 0
    f0, reg = find_f(x=x, R=[R0], p=np.ones(1), lr=lr_f, tol=tol, T=T_f)
    return f0, R0, reg


def algorithm(x, r, lr_r, lr_f, beta1, beta2,
              m, T_r, T_f,  avg_frac, stop_frac):

    regrets = np.zeros(m+1)
    f0, R0, reg = init_R0_f0(x, r, lr_f, tol=1e-3, T_f=T_f)
    regrets[0] = reg
    R = [R0]
    f = [f0]
    for i in range(m):
        new_R, p, o = find_R(x=x, R=R, f=f, T=T_r,
                             lr_r=lr_r, lr_f=lr_f, beta1=beta1, beta2=beta2,
                             avg_frac=avg_frac, stop_frac=stop_frac, tol=1e-3)
        R.append(new_R)
        new_f, reg = find_f(x=x, R=R, p=p, lr=lr_f, tol=1e-3, T=T_f)
        f.append(new_f)
        regrets[i+1] = reg
    _, p, o = find_R(x=x, R=R, f=f, T=T_r,
                     lr_r=lr_r, lr_f=lr_f, beta1=beta1, beta2=beta2,
                     avg_frac=avg_frac, stop_frac=stop_frac, tol=1e-3)
    return R, f, p, o, regrets

