import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from multi_lr_sol import plot_results



if __name__ == "__main__":
    
    for N in [100, 1_000, 10_000]:
        for runs in range(1, 11):
            for noise in [True, False]:
                try:
                    file_path = f"res_acc_mul_ll_N={N}_R={runs}"+"_Noise"*int(noise)+".p"
                    print(file_path)
                    with open(file_path, "rb") as file:
                        results = pickle.load(file)
                        plot_results(results=results, N=N, runs=runs, suffix="noise"*int(noise))
                except FileNotFoundError:
                    print(f"The file '{file_path}' does not exist.")
                    continue
                except Exception as e:
                    print("An error occuirred:", e)
                    exit()









