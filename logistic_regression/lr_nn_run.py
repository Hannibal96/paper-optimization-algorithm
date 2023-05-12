#%%

import matplotlib.pyplot as plt
from lr_nn_steps import *
import matplotlib.pyplot as plt
import pickle
import time
from lr_utils import plot_and_save

#%%

def run_lr_algo(d, r, b, sigma,
                lr_r, lr_f, beta_r, beta_f,
                m, avg_frac, stop_frac, T_r, T_f,
                times):
    x = torch.randn(b, 1, d) * sigma
    regrets = np.zeros([times, m+1])
    for t in range(times):
        start_time = time.time()
        print(f"#Exp={t}")
        R, f, p, o, regrets_t = algorithm(x=x, r=r,
                                       lr_r=lr_r, lr_f=lr_f, beta_f=beta_f, beta_r=beta_r,
                                       m=m, T_r=T_r, T_f=T_f, avg_frac=avg_frac, stop_frac=stop_frac)
        regrets[t, :] = regrets_t
        print(regrets_t[-1])
        run_time_seconds = time.time() - start_time
        print(f"Running Time: {run_time_seconds // 60}:{run_time_seconds % 60}")

    return regrets

#%%

b = 100
d = 15
r = 3
sigma=1

lr_r = 1e1
lr_f = 1e-1
beta_r = 0.9
beta_f = 0.9

m = 10
avg_frac=4
stop_frac=4
T_r=100
T_f=100

times = 3

#%%

regrets = run_lr_algo(d=d, r=r, b=b, sigma=sigma,
                      lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f,
                      m=m, avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f,
                      times=times)

#%%


