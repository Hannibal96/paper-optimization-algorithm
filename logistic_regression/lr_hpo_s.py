from lr_steps import *
import optuna
import argparse


def optimize(trial):

    beta_f = trial.suggest_float('beta_f', 0.5, 0.99)
    beta_r = trial.suggest_float('beta_r', 0.5, 0.99)
    lr_f = trial.suggest_float('lr_f', 0, 1.0)
    lr_r = trial.suggest_float('lr_r', 0, 1.0)
    stop_frac = trial.suggest_int('stop_frac', 1, 10)
    avg_frac = trial.suggest_int('avg_frac', stop_frac, 10)

    skew_s = np.linspace(0, 2, 5)
    res = np.zeros([len(skew_s), times])
    metric = 0
    for idx, s_skew in enumerate(skew_s):
        print(f"s_std = {s_skew}")
        basic_reg, s_reg = run_algorithm_S(b, d, r, m, sigma, s_skew, times, T_r, T_f,
                                           beta_f, beta_r, lr_f, lr_r, avg_frac_r=avg_frac, stop_frac_r=stop_frac)
        res[idx] = s_reg / basic_reg
        metric += sum(res)

    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize lr using S")
    parser.add_argument('--b', type=int, help="The batch size of the data", required=True)
    parser.add_argument('--d', type=int, help="The d value to test", required=True)
    parser.add_argument('--r', type=int, help="The r value to test", required=True)
    parser.add_argument('--m', type=int, help="The number of iterations of the algorithm", required=True)
    parser.add_argument('--times', type=int, help="The number of times to run the algorithm", required=True)
    parser.add_argument('--Tr', '--tr', type=int, help="The number of steps of the finding R phase", required=True)
    parser.add_argument('--Tf', '--tf', type=int, help="The number of steps of the finding F phase", required=True)
    parser.add_argument('--sigma', '--s', type=float, help="Sigma to test", required=True)

    parser.add_argument('--trials', type=int, help="The number of trials", required=False, default=100)

    args = parser.parse_args()

    b = args.b
    d = args.d
    r = args.r
    m = args.m
    T_r = args.Tr
    T_f = args.Tf
    times = args.times
    sigma = args.sigma

    name = f"opt-lr-s_b={b}_d={d}_r={r}_var={sigma}_m={m}_Tr={T_r}_Tf={T_f}"
    study = optuna.create_study(study_name=f'{name}',
                                storage=f'sqlite:///./../optuna/{name}.db ',
                                direction='minimize', load_if_exists=True)

    study.optimize(optimize, n_trials=args.trials)
