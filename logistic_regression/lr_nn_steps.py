import torch

from lr_nn_utils import *
import copy

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def find_Q(r, x, R, f, lr, tol=1e-4, T=100):
    Q = {}
    for j, r_nn in enumerate(R):
        for i, f_nn in enumerate(f):
            Q[(j, i)] = find_q(r=r, x=x, r_nn=r_nn, f_nn=f_nn, lr=lr, tol=tol, T=T)
    return Q


def find_f(lr, R, p, x, tol=1e-4, T=1000):
    d = R[0].linear_layer_1.weight.shape[0]
    r = R[0].linear_layer_2.weight.shape[0]

    f_nn = F_nn(d=d).to(device)
    optimizer = torch.optim.SGD(f_nn.parameters(), lr=lr)
    #prev_weights = f_nn.linear_layer.weight.clone()

    for t in range(T):
        Q = find_Q(r=r, x=x, R=R, f=[f_nn], lr=lr, tol=1e-4, T=100)
        f_loss = -regret(x=x, R=R, p=p, f=[f_nn], o=torch.ones(1), Q=Q)
        optimizer.zero_grad()
        f_loss.backward()
        optimizer.step()
        """with torch.no_grad():
            f_nn.linear_layer.weight /= torch.linalg.norm(f_nn.linear_layer.weight)
        if torch.linalg.norm(f_nn.linear_layer.weight - prev_weights) < tol:
            break
        else:
            prev_weights = f_nn.linear_layer.weight.clone()"""

    reg = regret(x=x, R=R, p=p, f=[f_nn], o=torch.ones(1), Q=Q)
    return f_nn, reg


def find_R(x, R, f,
           lr_r, lr_f, beta_f, beta_r,
           stop_frac, avg_frac, tol, T=100):

    R_pool = R[:]
    T_stop = T - T // stop_frac
    T_avg = T // avg_frac
    d = R[0].linear_layer_1.weight.shape[0]
    r = R[0].linear_layer_2.weight.shape[0]

    o = torch.ones([len(f), 1], requires_grad=False).to(device) / len(f)
    p = torch.ones([len(R_pool) + 1, 1], requires_grad=False).to(device) / (len(R_pool) + 1)

    o_sum = o * 0
    p_sum = p * 0

    r_nn = R_nn(d=d, r=r).to(device)
    R_pool.append(r_nn)
    optimizer = torch.optim.SGD(r_nn.parameters(), lr=lr_r)
    Q = find_Q(r=r, x=x, R=R_pool, f=f, lr=lr_f, tol=tol)
    for t in range(T):

        r_loss = regret(x=x, R=R_pool, p=p, f=f, o=o, Q=Q)
        optimizer.zero_grad()
        r_loss.backward()
        optimizer.step()

        R_pool[-1] = r_nn
        new_Q = find_Q(r=r, x=x, R=[r_nn], f=f, lr=lr_f, tol=tol, T=100)
        for i in range(len(f)):
            Q[(len(R_pool)-1, i)] = new_Q[(0, i)]

        with torch.no_grad():
            L = calc_loss_matrix(x=x, R=R_pool, Q=Q, f=f)
            if t < T_stop:
                o = MWU(p1=p, losses=-L, p2=o, beta=beta_f, dim=2)
                p = MWU(p1=p, losses=L, p2=o, beta=beta_r, dim=1)
                if t >= T_stop - T_avg:
                    o_sum = o_sum + o
                    p_sum = p_sum + p
            if t == T_stop:
                o = o_sum / sum(o_sum)
                p = p_sum / sum(p_sum)

    return r_nn, p, o


def init_R0_f0(x, r, lr_f, T_f, tol=1e-3):
    d = x[0].shape[1]
    R0 = R_nn(d=d, r=r).to(device)
    with torch.no_grad():
        R0.linear_layer_1.weight *= 0
        R0.linear_layer_2.weight *= 0
    f0, reg = find_f(lr=lr_f, R=[R0], p=torch.ones(1), x=x, tol=tol, T=T_f)
    return f0, R0, reg


def algorithm(x, r, lr_r, lr_f, beta_f, beta_r,
              m, T_r, T_f,  avg_frac, stop_frac):

    regrets = torch.zeros(m+1)
    f0, R0, reg = init_R0_f0(x, r, lr_f, tol=1e-3, T_f=T_f)
    regrets[0] = reg
    R = [R0]
    f = [f0]
    for i in range(m):
        print(f"m={i}")
        new_R, p, o = find_R(x=x, R=R, f=f, T=T_r,
                             lr_r=lr_r, lr_f=lr_f, beta_f=beta_f, beta_r=beta_r,
                             avg_frac=avg_frac, stop_frac=stop_frac, tol=1e-3)
        R.append(new_R)
        new_f, reg = find_f(x=x, R=R, p=p, lr=lr_f, tol=1e-3, T=T_f)
        f.append(new_f)
        regrets[i+1] = reg
    _, p, o = find_R(x=x, R=R, f=f, T=T_r,
                     lr_r=lr_r, lr_f=lr_f, beta_f=beta_f, beta_r=beta_r,
                     avg_frac=avg_frac, stop_frac=stop_frac, tol=1e-3)
    return R, f, p, o, regrets



















