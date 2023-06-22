import matplotlib.pyplot as plt
from nn_utils import *
import optuna
import argparse


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def optuna_optimization(trial):

    beta_f = trial.suggest_float('beta_f', 0.5, 0.99)
    beta_r = trial.suggest_float('beta1', 0.5, 0.99)
    lr_r = trial.suggest_float('lr_r', 0, 1.0)
    lr_f = trial.suggest_float('lr_f', 0, 1.0)
    stop_frac = trial.suggest_int('stop_frac', 2, 5)
    avg_frac = trial.suggest_int('avg_frac', 2, 5)

    regrets, regrets_mix = run_algorithm(d=d, r=r, m=m, times=times, d_sigma=d_sigma,
                                         beta_f=beta_f, beta_r=beta_r, T_r=T, lr_r=lr_r, lr_f=lr_f,
                                         avg_frac_r=avg_frac, stop_frac_r=stop_frac, opt_q=use_optimal_q)

    return sum(abs(np.log(regrets.min(axis=1) / regrets_mix)))


def plot_regrets(res, label, title=None, path=None):
    if title is None:
        plt.title(f"Regret vs Iteration")
    else:
        plt.title(title)
    plt.grid()
    plt.xlabel(f'#Iteration')
    plt.ylabel(f'Regret')
    plt.plot(res.T, label=label)
    plt.legend()
    if path is None:
        plt.show()
    else:
        plt.savefig(f"{path}_regrets.png")
        plt.clf()

    plt.plot(range(res.shape[1]), res.mean(axis=0), label=label)
    plt.fill_between(range(res.shape[1]),
                     res.mean(axis=0)+res.std(axis=0),
                     res.mean(axis=0)-res.std(axis=0),
                     color='r', alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel(f'#Iteration')
    plt.ylabel(f'Regret')
    if path is None:
        plt.show()
    else:
        plt.savefig(f"{path}_avg.png")
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN HPO")
    parser.add_argument('--d', type=int, help="The d value to test", required=True)
    parser.add_argument('--r', type=int, help="The r value to test", required=True)
    parser.add_argument('--sigma', "--s", type=int, help="The sigma value to test", required=True)
    parser.add_argument('--m', type=int, help="The number of iterations of the algorithm to use", required=True)
    parser.add_argument('--tr', "--Tr", type=int, help="The number of steps of the finding R phase", required=True)
    parser.add_argument('--times', type=int, help="The number of times to run the algorithm", required=True)
    parser.add_argument('--opt_q', '--optimal_q', help="Use optimal Q formula", required=False, action='store_true')

    args = parser.parse_args()

    d = args.d
    r = args.r
    T = args.tr
    m = args.m
    times = args.times
    d_sigma = args.sigma
    use_optimal_q = args.opt_q

    name = f"opt-nn_d={d}_r={r}_var={d_sigma}_m={m}_Tr={T}"
    study = optuna.create_study(study_name=f'{name}',
                                storage=f'sqlite:///./../optuna/{name}.db ',
                                direction='minimize', load_if_exists=True)
    study.optimize(optuna_optimization, n_trials=100)
