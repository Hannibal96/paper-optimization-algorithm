import numpy as np
import itertools
from mse.utils import matrix_power


def optimal_solution(sigma, r):
    ordered_eig = np.flip(np.sort(np.linalg.eigvals(sigma)))
    l_star = 0
    d = sigma.shape[0]
    for l in range(r, d + 1):
        if l == d:
            l_star = l
            break
        a = ordered_eig[l - 1] * sum((1 / ordered_eig)[0:l])
        b = ordered_eig[l] * sum((1 / ordered_eig)[0:l])
        if l >= r + b and l <= r + a:
            l_star = l
            break

    regret_mix = (l_star - r) / sum((1 / ordered_eig)[0:l_star])

    A = get_A(l_star=l_star, r=r)
    b = (1 / ordered_eig[0:l_star]) * regret_mix
    b = b.reshape(-1, 1)

    return l_star, regret_mix, A, b


def get_A(l_star, r):
    choose_list = range(l_star)
    combinations = list(itertools.combinations(choose_list, r))
    A = np.ones([len(combinations), l_star])
    for col_idx, a in enumerate(A):
        comb = combinations[col_idx]
        for idx in comb:
            a[idx] = 0
    return A.T


def get_R(d, r):
    choose_list = range(d)
    combinations = list(itertools.combinations(choose_list, r))
    R = []
    for comb in combinations:
        R_t = np.zeros([d, r])
        for i in range(r):
            R_t[comb[i]][i] = 1
        R.append(R_t)
    R = np.array(R)
    return R


def get_f(d):
    f = []
    for i in range(d):
        f_i = np.zeros([d,1])
        f_i[i] = 1
        f.append(f_i)
    return np.array(f)