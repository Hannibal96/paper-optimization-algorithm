from steps import *
from optimal import *
import optuna
import warnings
warnings.filterwarnings("error")
import argparse


def optuna_optimize_algorithm(trial):

    beta_f = trial.suggest_float('beta_f', 0.5, 0.99)
    beta_r = trial.suggest_float('beta_r', 0.5, 0.99)
    lr_f = trial.suggest_float('lr_f', 0, 1.0)
    lr_r = trial.suggest_float('lr_r', 0, 1.0)
    stop_frac = trial.suggest_int('stop_frac_r', 1, 10)
    avg_frac = trial.suggest_int('avg_frac_r', stop_frac, 10)

    regret_list, regret_mix_list = run_algorithm(d=d, r=r, m=m, sigma_type=sigma_type, s_type=s_type, times=times,
                                                 d_sigma=d_sigma, T_r=T_r, beta_f=beta_f, beta_r=beta_r, lr_f=lr_f, lr_r=lr_r,
                                                 avg_frac_r=avg_frac, stop_frac_r=stop_frac)

    return sum(abs(np.log(regret_list.min(axis=1) / regret_mix_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MSE linear setting")
    parser.add_argument('--d', type=int, help="The d value to test", required=True)
    parser.add_argument('--r', type=int, help="The r value to test", required=True)
    parser.add_argument('--m', type=int, help="The number of iterations of the algorithm", required=True)
    parser.add_argument('--T', '--Tr', type=int, help="The number of steps of the finding R phase", required=True)
    parser.add_argument('--times', type=int, help="The number of times to run the algorithm", required=True)
    parser.add_argument('--sigma', "--s", type=int, help="The sigma value to test", required=True)
    parser.add_argument('--trials', "-N", type=int, help="The number of trials of the HPO", required=True)
    args = parser.parse_args()

    d = args.d
    r = args.r
    m = args.m
    T_r = args.T
    times = args.times
    d_sigma = args.sigma

    sigma_type = matrix_type.DIAG
    s_type = matrix_type.IDENTITY

    name = f"opt-mse_d={d}_r={r}_sigma={sigma_type}_s={s_type}_m={m}_Tr={T_r}_var={d_sigma}"
    study = optuna.create_study(study_name=f'{name}',
                                storage=f'sqlite:///./../optuna/{name}.db ',
                                direction='minimize', load_if_exists=True)

    study.optimize(optuna_optimize_algorithm, n_trials=args.trials)
