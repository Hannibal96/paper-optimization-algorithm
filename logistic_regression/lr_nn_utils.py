import copy
import numpy as np
import torch
from torch import nn, optim
from lr_utils import MWU
import torch
import torch.nn.functional

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class F_nn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=d, out_features=1, bias=False)

    def forward(self, x: torch.Tensor):
        z = self.linear_layer(x)
        return torch.sigmoid(z)


class Q_nn(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=r, out_features=1, bias=False)

    def forward(self, z: torch.Tensor):
        z = self.linear_layer(z)
        return torch.sigmoid(z)


class R_nn(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=d, out_features=r, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear_layer(x)


def loss(x, r_nn, f_nn, q_nn):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    P = f_nn(x)
    Q = q_nn(r_nn(x))
    P = torch.concat([P.reshape(-1, 1), 1 - P.reshape(-1, 1)], dim=1)
    Q = torch.concat([Q.reshape(-1, 1), 1 - Q.reshape(-1, 1)], dim=1)
    return kl_loss(torch.log(Q), P)


def regret(x, R, p, f, o, Q):
    regret = 0
    for j, R_j in enumerate(R):
        for i, f_i in enumerate(f):
            q_ji = Q[(j, i)]
            regret += p[j] * o[i] * loss(x=x, r_nn=R_j, f_nn=f_i, q_nn=q_ji)
    return regret


def find_q(r, x, r_nn, f_nn, lr, tol=1e-4, T=1_000_000):
    q_nn = Q_nn(r=r).to(device)
    optimizer = torch.optim.SGD(q_nn.parameters(), lr=lr)
    for t in range(T):
        q_loss = loss(x=x, r_nn=r_nn, q_nn=q_nn, f_nn=f_nn)
        q_nn.zero_grad()
        q_loss.backward()
        optimizer.step()
        if torch.linalg.norm(q_nn.linear_layer.weight.grad) < tol:
            break
    return q_nn


def calc_loss_matrix(x, R, Q, f):
    losses_matrix = torch.zeros([len(R), len(f)]).to(device)
    for j in range(len(R)):
        R_j = R[j]
        for i in range(len(f)):
            f_i = f[i]
            q_ji = Q[(j, i)]
            losses_matrix[j, i] = loss(x=x, r_nn=R_j, f_nn=f_i, q_nn=q_ji)
    return losses_matrix





