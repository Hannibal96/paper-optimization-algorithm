import copy
import numpy as np
import torch
from torch import nn, optim

device = 'cpu'


class F(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=d, out_features=1, bias=False)

    def forward(self, x: torch.Tensor):
        z = self.linear_layer(x)
        return torch.sigmoid(z)


class Q(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=r, out_features=1, bias=False)

    def forward(self, z: torch.Tensor):
        z = self.linear_layer(z)
        return torch.sigmoid(z)


class R(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=d, out_features=r, bias=False)

    def forward(self, z: torch.Tensor):
        return self.linear_layer(x)


f = F(in_features=4)
x = torch.randn(1, 4)

f(x)

print('hi')



