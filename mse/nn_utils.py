import copy

import numpy as np
import torch
from torch import nn, optim
from utils import MWU
from tqdm import tqdm
from optimal import optimal_solution

device = 'cpu'


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=in_features,
                                      out_features=out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def sanity_check_pg_f(sigma, f_n, p, R, Q):
    with torch.no_grad():
        f = f_n.linear_layer.weight.T
        sum_rep_pred = f * 0
        for j in range(len(R)):
            R_j = R[j].linear_layer.weight.T
            p_j = p.T[j]
            q_ji = Q[(j, 0)].linear_layer.weight.T
            sum_rep_pred = sum_rep_pred + p_j * R_j @ q_ji
        pg_f = sigma @ (f - sum_rep_pred)
        return (pg_f / f_n.linear_layer.weight.grad.T).std()


def sanity_check_pg_R(sigma, R_n, f, o, Q):
    with torch.no_grad():
        R = R_n.linear_layer.weight.T
        sum_projection = R * 0
        for i in range(len(f)):
            f_i = f[i].linear_layer.weight.T
            q_ji = Q[(0, i)].linear_layer.weight.T
            o_i = o[i]
            sum_projection = sum_projection + o_i * (R @ q_ji - f_i) @ q_ji.T
        pg_R = sigma @ sum_projection
        return (pg_R / R_n.linear_layer.weight.grad.T).std()


def calc_loss(sigma, f_i, R_j, q_ji):
    return f_i(f_i(sigma).T) + q_ji(R_j(q_ji(R_j(sigma)).T)) - 2 * f_i(q_ji(R_j(sigma)).T)


def calc_loss_matrix(sigma, R, Q, f):
    losses_matrix = torch.zeros([len(R), len(f)]).to(device)
    for j in range(len(R)):
        for i in range(len(f)):
            losses_matrix[j, i] = calc_loss(sigma=sigma, R_j=R[j], f_i=f[i], q_ji=Q[(j, i)])
    return losses_matrix


def calc_regret(loss_matrix, p, o):
    return p @ loss_matrix @ o


def optimal_q_formula(sigma, R_j, f_i):
    inv = R_j.linear_layer.weight @ sigma @ R_j.linear_layer.weight.T
    if R_j.linear_layer.weight.norm() == 0:
        return inv * 0 @ R_j.linear_layer.weight @ sigma @ f_i.linear_layer.weight.T
    return torch.linalg.inv(inv) @ R_j.linear_layer.weight @ sigma @ f_i.linear_layer.weight.T


def optimal_q(sigma, R_j, f_i, tol, lr=1e-3, use_formula=False):
    r = R_j.linear_layer.weight.shape[0]
    q = LinearModel(in_features=r, out_features=1).to(device=device)

    if use_formula:
        with torch.no_grad():
            q.linear_layer.weight = nn.Parameter(optimal_q_formula(sigma=sigma, R_j=R_j, f_i=f_i).T)
        return q

    prev_weights = q.linear_layer.weight.clone()

    optimizer = optim.SGD(q.parameters(), lr=lr)
    for t in range(10000):
        optimizer.zero_grad()
        loss = calc_loss(sigma=sigma, f_i=f_i, R_j=R_j, q_ji=q)
        loss.backward()
        optimizer.step()
        if torch.linalg.norm(q.linear_layer.weight.grad) / lr < tol \
                or torch.linalg.norm(q.linear_layer.weight - prev_weights) / lr < tol:
            break
        else:
            prev_weights = q.linear_layer.weight.clone()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            q = LinearModel(in_features=r, out_features=1).to(device=device)
            lr = lr / 2
            optimizer = optim.SGD(q.parameters(), lr=lr)

    return q


def optimal_Q(sigma, R, f, use_opt_q, tol=1e-1):
    Q = {}
    for j in range(len(R)):
        for i in range(len(f)):
            Q[(j, i)] = optimal_q(sigma=sigma, R_j=R[j], f_i=f[i], tol=tol, use_formula=use_opt_q)
    return Q


def find_worst_f(sigma, R, p, lr, opt_q, tol=1e-2):
    d = sigma.shape[0]
    f = LinearModel(in_features=d, out_features=1).to(device=device)
    prev_f_weights = f.linear_layer.weight.clone()
    optimizer = optim.SGD(f.parameters(), lr=lr)

    counter = 0
    while True:
        Q = optimal_Q(sigma=sigma, R=R, f=[f], use_opt_q=opt_q)

        optimizer.zero_grad()
        loss_matrix = calc_loss_matrix(sigma=sigma, R=R, Q=Q, f=[f])
        loss = -calc_regret(loss_matrix=loss_matrix, p=p, o=torch.ones(1).to(device))

        loss.backward()
        """if sanity_check_pg_f(sigma=sigma, R=R, f_n=f, p=p, Q=Q) > 1e-3:
            print(sanity_check_pg_f(sigma=sigma, R=R, f_n=f, p=p, Q=Q))"""
        optimizer.step()

        with torch.no_grad():
            f.linear_layer.weight.div_(torch.norm(f.linear_layer.weight, dim=1, keepdim=True))

        if torch.linalg.norm(f.linear_layer.weight - prev_f_weights)/lr < tol:
            break
        else:
            prev_f_weights = f.linear_layer.weight.clone()
        counter += 1
        if counter > 10000:
            break

    loss_matrix = calc_loss_matrix(sigma=sigma, R=R, f=[f], Q=optimal_Q(sigma=sigma, R=R, f=[f], use_opt_q=opt_q))
    regret = calc_regret(loss_matrix=loss_matrix, p=p, o=torch.ones(1))
    return f, regret


def find_new_R(sigma, R, f, T, d, r, beta1, beta2, avg_frac, stop_frac, lr, opt_q):
    R_pool = R[:]
    T_stop = T - T // stop_frac
    T_avg = T // avg_frac

    o = torch.ones([len(f), 1]).to(device) / len(f)
    p = torch.ones([len(R_pool)+1, 1]).to(device) / (len(R_pool)+1)

    o_sum = o * 0
    p_sum = p * 0

    new_R = LinearModel(in_features=d, out_features=r).to(device=device)
    optimizer = optim.SGD(new_R.parameters(), lr=lr)
    R_pool.append(new_R)
    Q = optimal_Q(sigma=sigma, R=R_pool, f=f, use_opt_q=opt_q)  #

    for t in tqdm(range(T)):
        #Q = optimal_Q(sigma=sigma, R=[new_R], f=f)

        optimizer.zero_grad()
        loss_matrix = calc_loss_matrix(sigma=sigma, R=[new_R], Q=Q, f=f)
        loss = calc_regret(loss_matrix=loss_matrix, p=torch.ones(1).to(device), o=o)
        loss.backward()
        """if sanity_check_pg_R(sigma=sigma, R_n=new_R, f=f, o=o, Q=Q) > 1e-3:
            print(sanity_check_pg_R(sigma=sigma, R_n=new_R, f=f, o=o, Q=Q))"""
        optimizer.step()

        R_pool[-1] = new_R

        #Q = optimal_Q(sigma=sigma, R=R_pool, f=f)
        new_Q = optimal_Q(sigma=sigma, R=[new_R], f=f, use_opt_q=opt_q)  #
        for i in range(len(f)): #
            Q[(0, i)] = copy.deepcopy(new_Q[(0, i)]) #

        L = calc_loss_matrix(sigma=sigma, R=R_pool, Q=Q, f=f)

        with torch.no_grad():
            if t < T_stop:
                o = MWU(p1=p, losses=-L, p2=o, beta=beta1, dim=2)
                p = MWU(p1=p, losses=L, p2=o, beta=beta2, dim=1)
                if t >= T_stop - T_avg:
                    o_sum = o_sum + o
                    p_sum = p_sum + p
            if t == T_stop:
                o = o_sum / sum(o_sum)
                p = p_sum / sum(p_sum)

    return new_R, p.T, o


def init_R0_f0(sigma, d, r, T,
               beta_r, beta_f, lr_r, lr_f,
               avg_frac_r, stop_frac_r, opt_q):

    R0 = LinearModel(in_features=d, out_features=1).to(device=device)
    with torch.no_grad():
        R0.linear_layer.weight.mul_(0)
    f0, reg = find_worst_f(sigma=sigma, R=[R0], p=torch.ones(1).to(device), lr=lr_f, opt_q=opt_q)
    f = [f0]
    R = [R0]
    for i in range(r):
        new_R, p, o = find_new_R(sigma=sigma, R=R, f=f, d=d, r=1, T=T, lr=lr_r,
                                 beta1=beta_f, beta2=beta_r,
                                 avg_frac=avg_frac_r, stop_frac=stop_frac_r, opt_q=opt_q)
        R.append(new_R)
        new_f, reg = find_worst_f(sigma=sigma, R=R, p=p, lr=lr_f, opt_q=opt_q)
        f.append(new_f)

    R0 = LinearModel(in_features=d, out_features=r).to(device=device)
    with torch.no_grad():
        for i in range(r):
            R0.linear_layer.weight[i] = R[i+1].linear_layer.weight

    return R0, f, reg


def algorithm(sigma, m, r, beta_f, beta_r, T_r, lr_r, lr_f, avg_frac_r, stop_frac_r, opt_q):

    regrets = np.zeros(m+1)
    d = sigma.shape[0]

    R0, f0, regret = init_R0_f0(sigma=sigma, d=d, r=r, T=T_r,
                                beta_r=beta_r, beta_f=beta_f, lr_r=lr_r, lr_f=lr_f,
                                avg_frac_r=avg_frac_r, stop_frac_r=stop_frac_r, opt_q=opt_q)

    f = f0[:]
    R = [R0]
    p = torch.ones(1).to(device)
    o = torch.ones(len(f)).to(device) / len(f)

    new_f, regret = find_worst_f(sigma=sigma, R=R, p=p, lr=lr_f, opt_q=opt_q)
    regrets[0] = regret

    for i in range(m):
        new_R, p, o = find_new_R(sigma=sigma, R=R, f=f, T=T_r, d=d, r=r,
                                 beta1=beta_f, beta2=beta_r, avg_frac=avg_frac_r, stop_frac=stop_frac_r, lr=lr_r, opt_q=opt_q)
        R.append(new_R)

        new_f, regret = find_worst_f(sigma, R, p, lr=lr_f, opt_q=opt_q)
        regrets[i+1] = regret
        f.append(new_f)

    return R, f, regrets


def run_algorithm(d, r, m, times, beta_f, beta_r, T_r, lr_r, lr_f, avg_frac_r, stop_frac_r, d_sigma, opt_q):
    regrets = np.zeros([times, m+1])
    regrets_mix = np.zeros(times)

    for t in range(times):
        D, _ = torch.sort(torch.randn(d) * d_sigma)
        sigma = torch.diag(torch.exp(torch.flip(D, dims=(0,)))).to(device)

        l_star, regret_mix, A, b = optimal_solution(sigma=sigma, r=r)
        print(f"l_star={l_star}", end=" ; ")
        print(f"regret_mix={regret_mix}", end=" ; ")
        regrets_mix[t] = regret_mix

        R, f, curr_regrets = algorithm(sigma=sigma, m=m, r=r,
                                       beta_f=beta_f, beta_r=beta_r, lr_r=lr_r, lr_f=lr_f,
                                       T_r=T_r, avg_frac_r=avg_frac_r, stop_frac_r=stop_frac_r, opt_q=opt_q)
        regrets[t, :] = curr_regrets
        print(f"algo_regret={curr_regrets.min()}")

    return regrets, regrets_mix

