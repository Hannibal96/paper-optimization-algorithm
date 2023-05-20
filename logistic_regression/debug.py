#%%

import matplotlib.pyplot as plt
import numpy as np

from lr_steps import *
import matplotlib.pyplot as plt
import pickle
import time
from lr_utils import plot_and_save

#%%

b = 1000
d = 5
r = 2
sigma=1

lr_r = 1e-3
lr_f = 1e-1
beta_r = 0.9
beta_f = 0.9

m = 5
avg_frac=4
stop_frac=4
T_r = 100
T_f = 10_000

times = 10

#%%

skew_s = np.linspace(0, 2, 5)
res = np.zeros([len(skew_s), times])

for idx, s_skew in enumerate(skew_s):
    print(f"s_std = {s_skew}")
    basic_reg, s_reg = run_algorithm_S(b, d, r, m, sigma, s_skew, times, T_r, T_f,
                                       beta_f, beta_r, lr_f, lr_r, avg_frac_r=4, stop_frac_r=4)
    res[idx] = basic_reg / s_reg

#%%

#%%


