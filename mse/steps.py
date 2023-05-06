import numpy as np

from utils import *
from tqdm import tqdm
from optimal import optimal_solution


def project_f(f, S):
    return f / np.linalg.norm(matrix_power(A=S, p=-0.5) @ f)


def find_final_f(sigma, S, R, p, lr, tol=1e-4):
    d = sigma.shape[0]
    f_new = np.random.randn(d, 1)
    f_new = project_f(f=f_new, S=S)
    f_prev = f_new.copy()

    while True:
        Q = optimal_Q(sigma=sigma, R=R, f=[f_new])
        pg_f = partial_grad_f(sigma=sigma, f_i=f_new, R=R, p=p, Q_i=Q[:, 0])
        f_new = f_new + lr * pg_f
        f_new = project_f(f=f_new, S=S)
        if np.linalg.norm(f_new - f_prev) < tol:
            break
        else:
            f_prev = f_new.copy()

    regret = calc_regret(sigma=sigma, R=R, f=[f_new],
                         Q=optimal_Q(sigma=sigma, R=R, f=[f_new]), p=p, o=np.ones(1))
    return f_new, p, regret


def find_new_f(sigma, S, R, beta, lr, T, stop_frac, avg_frac):

    T_stop = T // stop_frac
    T_avg = T // avg_frac

    p = np.ones(len(R)) / len(R)
    p_sum = p * 0

    d = sigma.shape[0]
    f_new = np.random.randn(d, 1)
    f_new = project_f(f=f_new, S=S)
    f = [f_new]

    for t in range(T):
        Q = optimal_Q(sigma=sigma, R=R, f=f)
        pg_f = partial_grad_f(sigma=sigma, f_i=f_new, R=R, p=p, Q_i=Q[:, 0])
        f_new = f_new + lr * pg_f
        f_new = project_f(f=f_new, S=S)

        f = [f_new]

        losses = calc_loss_matrix(sigma=sigma, R=R, Q=optimal_Q(sigma=sigma, R=R, f=f), f=f)
        if t < T_stop:
            p = MWU(p1=p, losses=losses, p2=np.ones([1]), beta=beta, dim=1)
            if t > T_stop - T_avg:
                p_sum += p
        if t == T_stop:
            p = p_sum / sum(p_sum)

    regret = calc_regret(sigma=sigma, R=R, f=f, Q=optimal_Q(sigma=sigma, R=R, f=f), p=p, o=np.ones(1))
    return f_new, p, regret


def find_new_R(sigma, R, f, T, beta_f, beta_r, avg_frac, stop_frac, lr):

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

    for t in range(T):
        Q = optimal_Q(sigma=sigma, R=[new_R], f=f)
        new_R = new_R - lr * partial_grad_R(sigma=sigma, R_j=new_R, f=f, o=o, Q_j=Q[0, :])

        R_pool[-1] = new_R

        Q = optimal_Q(sigma=sigma, R=R_pool, f=f)
        L = calc_loss_matrix(sigma=sigma, R=R_pool, Q=Q, f=f)

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
                p = np.ones([len(R_pool), 1]) / len(R_pool)

    return new_R, p, o


def init_R0_f0(sigma, S, beta_f, beta_r, T_r, r,
               lr_f, lr_r, avg_frac_r, stop_frac_r):
    d = sigma.shape[0]
    f0 = []
    R0 = [np.zeros([d, 1])]
    for i in range(r):
        new_f, p, _ = find_final_f(sigma=sigma, S=S, R=R0, p=np.ones(len(R0)) / len(R0), lr=lr_f)
        new_f = np.round(new_f, 3)
        f0.append(new_f)

        new_R, p, o = find_new_R(sigma=sigma, R=R0, f=f0, T=T_r, beta_f=beta_f,
                                 beta_r=beta_r, stop_frac=stop_frac_r, avg_frac=avg_frac_r, lr=lr_r)
        R0.append(new_R)

    _, _, regret = find_final_f(sigma=sigma, S=S, R=R0, p=np.ones(len(R0)) / len(R0), lr=lr_f)
    return np.array(R0[1:]).T[0], f0, regret


def algorithm(sigma, S, m, r, beta_f, beta_r, T_r, lr_f, lr_r, avg_frac_r, stop_frac_r):

    regrets = np.zeros(m+2)
    R0, f0, regret = init_R0_f0(sigma=sigma, S=S, T_r=T_r, r=r,
                                beta_f=beta_f, beta_r=beta_r,
                                lr_f=lr_f, lr_r=lr_r,
                                avg_frac_r=avg_frac_r, stop_frac_r=stop_frac_r)
    regrets[0] = regret

    f = f0[:]
    R = [R0]
    p = np.ones(1)

    for i in range(m):
        new_f, p, regret = find_final_f(sigma=sigma, S=S, R=R, p=p, lr=lr_r)

        regrets[i+1] = regret
        new_f = np.round(new_f, 3)
        f.append(new_f)

        new_R, p, o = find_new_R(sigma=sigma, R=R, f=f, T=T_r,
                                 lr=lr_r, beta_r=beta_r, beta_f=beta_f,
                                 stop_frac=stop_frac_r, avg_frac=avg_frac_r)
        new_R = np.real(new_R)
        R.append(new_R)

    _, _, final_regret = find_final_f(sigma=sigma, S=S, R=R, p=p, lr=lr_f)
    regrets[m+1] = final_regret

    return R, f, regrets


def run_algorithm(d, r, m, sigma_type, s_type, times, d_sigma, T_r,
                  beta_f, beta_r, lr_f, lr_r, avg_frac_r, stop_frac_r):

    regret_list = np.zeros([times, m+2])
    regret_mix_list = np.zeros(times)

    for i in range(times):

        sigma = get_pdm(d=d, type=sigma_type) ** d_sigma
        S = get_pdm(d=d, type=s_type)

        S_half = matrix_power(A=S, p=0.5)
        sigma = S_half @ sigma @ S_half

        l_star, regret_mix, A, b = optimal_solution(sigma=sigma, r=r)
        print(f"l_star={l_star}", end=" ; ")
        print(f"regret_mix={regret_mix}", end=" ; ")

        R, f, regrets = algorithm(sigma=sigma, S=get_pdm(d=d, type=matrix_type.IDENTITY), m=m, r=r, T_r=T_r,
                                  beta_f=beta_f, beta_r=beta_r, lr_f=lr_f, lr_r=lr_r,
                                  avg_frac_r=avg_frac_r, stop_frac_r=stop_frac_r)

        regret_list[i, :] = np.array(regrets)
        print(f"last_regret={min(regrets)}")
        regret_mix_list[i] = np.real(regret_mix)

    return regret_list, regret_mix_list