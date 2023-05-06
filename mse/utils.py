import numpy as np
from enum import Enum


def optimal_q(sigma, R_j, f_i):
    if (R_j == 0).all():
        r = R_j.shape[1]
        return np.zeros([r, 1])

    in_inv = R_j.T @ sigma @ R_j
    if np.linalg.matrix_rank(in_inv) < R_j.shape[1]:
        in_inv = in_inv + 1e-9 * np.trace(in_inv) * np.eye(R_j.shape[1])

    inv = np.linalg.inv(in_inv)
    q_ij = inv @ R_j.T @ sigma @ f_i
    return q_ij


def optimal_Q(sigma, R, f):
    Q = np.zeros([len(R), len(f), R[0].shape[1], 1])
    for j in range(len(R)):
        R_j = R[j]
        for i in range(len(f)):
            f_i = f[i]
            Q[j, i] = optimal_q(sigma=sigma, R_j=R_j, f_i=f_i)
    return Q


def partial_grad_f(sigma, f_i, R, p, Q_i):
    sum_rep_pred = f_i * 0
    for j in range(len(R)):
        R_j = R[j]
        p_j = p[j]
        q_ji = Q_i[j]
        sum_rep_pred = sum_rep_pred + p_j * R_j @ q_ji

    return sigma @ (f_i - sum_rep_pred)


def partial_grad_R(sigma, R_j, f, o, Q_j):
    sum_projection = R_j * 0
    for i in range(len(f)):
        f_i = f[i]
        q_ji = Q_j[i]
        o_i = o[i]
        sum_projection = sum_projection + o_i * (R_j @ q_ji - f_i) @ q_ji.T
    return sigma @ sum_projection


def calc_loss(sigma, R_j, f_i, q_ji):
    return f_i.T @ sigma @ f_i + q_ji.T @ R_j.T @ sigma @ R_j @ q_ji - 2 * f_i.T @ sigma @ R_j @ q_ji


def calc_loss_matrix(sigma, R, Q, f):
    losses_matrix = np.zeros([len(R), len(f)])
    for j in range(len(R)):
        R_j = R[j]
        for i in range(len(f)):
            f_i = f[i]
            q_ji = Q[j, i]
            losses_matrix[j, i] = calc_loss(sigma=sigma, R_j=R_j, f_i=f_i, q_ji=q_ji)
    return losses_matrix


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


def get_R_PCA(R):
    lamda, u = np.linalg.eig(R @ np.linalg.inv(R.T @ R) @ R.T)
    return u[:, lamda > 0.1]


def calc_regret(sigma, R, f, Q, p, o):
    regret = 0
    for j in range(len(R)):
        R_j = R[j]
        p_j = p[j]
        for i in range(len(f)):
            o_i = o[i]
            f_i = f[i]
            q_ji = Q[j, i]
            regret = regret + p_j * o_i * calc_loss(sigma=sigma, R_j=R_j, f_i=f_i, q_ji=q_ji)

    return regret


def matrix_power(A, p):
    D, U = np.linalg.eig(A)
    return U @ (np.diag(D**p)) @ U.T


class matrix_type(Enum):
    IDENTITY = 0
    DIAG = 1
    DIAG_NORM = 2
    NORM = 3
    FREE = 4


def get_pdm(d, type):
    if type == matrix_type.IDENTITY:
        return np.eye(d)
    if type == matrix_type.DIAG or type == matrix_type.DIAG_NORM:
        D = np.flip(np.sort(np.exp(np.random.randn(d))))
        S = np.diag(D)
        if type == matrix_type.DIAG_NORM:
            S = (S / D.sum()) * d
        return S
    if type == matrix_type.FREE or type == matrix_type.NORM:
        s = np.random.randn(d, d)
        S = s @ s.T
        if type == matrix_type.NORM:
            S = d * S / np.trace(S)
        return S


