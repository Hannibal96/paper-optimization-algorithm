from steps import *
from optimal import *
import matplotlib.pyplot as plt
import optuna
import pickle
import argparse
#import warnings
#warnings.filterwarnings("error")


def plot_res(x_axis, results, x_label, y_label, label, title, path):

    plt.plot(x_axis, results.mean(axis=1), label=label)
    plt.fill_between(x_axis, results.mean(axis=1)+results.std(axis=1), results.mean(axis=1)-results.std(axis=1),
                     color='r', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.savefig(path+".png")
    plt.clf()

    plt.grid()
    plt.boxplot(results.T, positions=x_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig("BP-"+path+".png")
    plt.clf()


def test_sigma(study, opt_d, opt_r, opt_sigma,
               d, r, T, times, m,
               sigma_max, sigma_min, spaces,
               sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False):

    running_params = f"sigma = {sigma_min} - {sigma_max} - {spaces}"
    opt_params = f"opt-[d, r, sigma]=[{opt_d}, {opt_r}, {opt_sigma}]"
    problem_params = f"Tr={T}_[d, r]=[{d}, {r}]"
    sigma_suffix = f'{running_params}_{opt_params}_{problem_params}'
    params = study.best_params

    if not load:
        d_sigmas = np.linspace(sigma_min, sigma_max, int((sigma_max - sigma_min) / spaces) + 1)
        results = np.zeros([len(d_sigmas), times])
        for idx, d_sigma in enumerate(d_sigmas):
            print(f" ***** Sigma={d_sigma} *****")
            regret_list, regret_mix_list = run_algorithm(d=d, r=r, m=m, sigma_type=sigma_type, s_type=s_type,
                                                         times=times, d_sigma=d_sigma, T_r=T,
                                                         **params)
            ratio = regret_list.min(axis=1) / regret_mix_list
            results[idx, :] = ratio

        with open(f'acc-ratio_vs_sigma0_{sigma_suffix}.p', 'wb') as f:
            pickle.dump(results, f)
        with open(f'sigma0-acc_{sigma_suffix}.p', 'wb') as f:
            pickle.dump(d_sigmas, f)

    else:
        with open(f'acc-ratio_vs_sigma0_{sigma_suffix}.p', 'rb') as f:
            results = pickle.load(f)
        with open(f'sigma0-acc_{sigma_suffix}.p', 'rb') as f:
            d_sigmas = pickle.load(f)

    plot_res(x_axis=d_sigmas, results=results,
             x_label=r'$\sigma_0$', y_label='Accuracy Ratio', label=f'd={d}, r={r}',
             title=r'results accuracy ratio vs $\sigma_0$',
             path=f"results_accuracy_ratio_sigma0_{sigma_suffix}")


def test_r(study, opt_d, opt_r, opt_sigma,
           d, sigma, T, times, m,
           r_max, r_min, spaces,
           sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False):
    r_min, r_max, spaces = int(r_min), int(r_max), int(spaces)

    running_params = f"r = {r_min} - {r_max} - {spaces}"
    opt_params = f"opt-[d, r, sigma]=[{opt_d}, {opt_r}, {opt_sigma}]"
    problem_params = f"Tr={T}_[d, sigma]=[{d}, {sigma}]"
    r_suffix = f'{running_params}_{opt_params}_{problem_params}'
    params = study.best_params

    if not load:
        r_list = range(r_min, r_max, spaces)
        results = np.zeros([len(r_list), times])
        for idx, r in enumerate(r_list):
            regret_list, regret_mix_list = run_algorithm(d=d, r=r, m=m, sigma_type=sigma_type, s_type=s_type,
                                                         times=times, d_sigma=sigma, T_r=T_r,
                                                         **params)
            ratio = regret_list.min(axis=1) / regret_mix_list
            results[idx, :] = ratio

        with open(f'acc-ratio_vs_r_.p', 'wb') as f:
            pickle.dump(results, f)
        with open(f'r-acc_{r_suffix}.p', 'wb') as f:
            pickle.dump(r_list, f)

    else:
        with open(f'acc-ratio_vs_r_{r_suffix}.p', 'rb') as f:
            results = pickle.load(f)
        with open(f'r-acc_{r_suffix}.p', 'rb') as f:
            r_list = pickle.load(f)

    plot_res(x_axis=r_list, results=results,
             x_label='r', y_label='Accuracy Ratio', label=f"d={d}" + r" $\sigma_0$=" + f"{opt_sigma}",
             title='results accuracy ratio vs r',
             path=f"results_accuracy_ratio_r_{r_suffix}")


def test_d(study, opt_d, opt_r, opt_sigma,
           r, sigma, T, times, m,
           d_max, d_min, spaces,
           sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False):

    d_min, d_max, spaces = int(d_min), int(d_max), int(spaces)

    running_params = f"d = {d_min} - {d_max} - {spaces}"
    opt_params = f"opt-[d, r, sigma]=[{opt_d}, {opt_r}, {opt_sigma}]"
    problem_params = f"Tr={T}_[r, sigma]=[{r}, {sigma}]"
    d_suffix = f'{running_params}_{opt_params}_{problem_params}'
    params = study.best_params

    if not load:
        d_list = range(d_min, d_max, spaces)
        results = np.zeros([len(d_list), times])

        for idx, d in enumerate(d_list):
            regret_list, regret_mix_list = run_algorithm(d=d, r=opt_r, m=m, sigma_type=sigma_type, s_type=s_type,
                                                         times=times, d_sigma=opt_sigma, T_r=T_r,
                                                         **params)
            ratio = regret_list.min(axis=1) / regret_mix_list
            results[idx, :] = ratio

        with open(f'acc-ratio_vs_d_{d_suffix}.p', 'wb') as f:
            pickle.dump(results, f)
        with open(f'd-acc_{d_suffix}.p', 'wb') as f:
            pickle.dump(d_list, f)

    else:
        with open(f'acc-ratio_vs_d_{d_suffix}.p', 'rb') as f:
            results = pickle.load(f)
        with open(f'd-acc_{d_suffix}.p', 'rb') as f:
            d_list = pickle.load(f)

    plot_res(x_axis=d_list, results=results,
             x_label='d', y_label='Accuracy Ratio', label=f"r={opt_r}" + r" $\sigma_0$=" + f"{opt_sigma}",
             title='accuracy ratio vs d',
             path=f"accuracy_ratio_d_{d_suffix}")


def test_m(study, opt_d, opt_r, opt_sigma,
           d, m, r, sigma, times, T,
           sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False):

    opt_params = f"opt-[d, r, sigma]=[{opt_d}, {opt_r}, {opt_sigma}]"
    problem_params = f"Tr={T}_[d, r, sigma]=[{d}, {r}, {sigma}]"
    m_suffix = f'{opt_params}_{problem_params}'
    params = study.best_params

    if not load:
        regret_list, regret_mix_list = run_algorithm(d=d, r=r, m=m, sigma_type=sigma_type, s_type=s_type,
                                                     times=times, d_sigma=opt_sigma, T_r=T,
                                                     **params)
        with open(f'regret_vs_m{m_suffix}.p', 'wb') as f:
            pickle.dump(regret_list, f)

    else:
        with open(f'regret_vs_m{m_suffix}.p', 'rb') as f:
            regret_list = pickle.load(f)

    plot_res(x_axis=range(m + 2), results=regret_list.T,
             x_label='#Iteration', y_label='Regret', label=f'd={d} r={r} ' + r'$\sigma_0$=' + f'{sigma}',
             title='Regret vs #Iteration',
             path=f"Regret_vs_Iteration_{m_suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MSE linear setting")
    parser.add_argument('--opt_d', type=int, help="Dimension of the vector", required=True)
    parser.add_argument('--opt_r', type=int, help="The r value of the optuna HPO", required=True)
    parser.add_argument('--opt_s', "--opt_sigma", type=int, help="The sigma value of the optuna HPO", required=True)
    parser.add_argument('--opt_m', type=int, help="The number of iterations of the algorithm HPO", required=True)
    parser.add_argument('--opt_tr', "--opt_Tr", type=int, help="The number of steps of the finding R phase", required=True)
    parser.add_argument('--opt_times', type=int, help="The number of times to run the algorithm", required=False)

    parser.add_argument('--d', type=int, help="The d value to test", required=False)
    parser.add_argument('--r', type=int, help="The r value to test", required=False)
    parser.add_argument('--sigma', "--s", type=int, help="The sigma value to test", required=False)
    parser.add_argument('--m', type=int, help="The number of iterations of the algorithm to use", required=False)
    parser.add_argument('--tr', "--Tr", type=int, help="The number of steps of the finding R phase", required=False)
    parser.add_argument('--times', type=int, help="The times value of the optuna HPO", required=False)

    parser.add_argument('--test', "--Test", "--TEST", type=str, help="The type of testing to run", choices=["sigma", "r", "m", "d"], required=True)
    parser.add_argument('--min', type=float, help="The min value of the range to test")
    parser.add_argument('--max', type=float, help="The max value of the range to test")
    parser.add_argument('--spaces', '--steps', type=float, help="The space interval of the range to test")

    args = parser.parse_args()

    opt_d = args.opt_d
    opt_r = args.opt_r
    opt_sigma = args.opt_s
    opt_m = args.opt_m
    opt_tr = args.opt_tr
    opt_times = args.opt_times

    d = args.d if args.d else opt_d
    r = args.r if args.r else opt_r
    sigma = args.sigma if args.sigma else opt_sigma
    m = args.m if args.m else opt_m
    T_r = args.tr if args.tr else opt_tr
    times = args.times if args.times else opt_times

    sigma_type = matrix_type.DIAG
    s_type = matrix_type.IDENTITY

    name = f"opt-mse_d={opt_d}_r={opt_r}_var={opt_sigma}_sigma={sigma_type}_s={s_type}_m={opt_m}_Tr={opt_tr}_times={opt_times}"
    study = optuna.create_study(study_name=f'{name}',
                                storage=f'sqlite:///./../optuna/{name}.db ',
                                direction='minimize', load_if_exists=True)

    if args.test == "sigma":
        test_sigma(study=study, opt_d=opt_d, opt_r=opt_r, opt_sigma=opt_sigma,
                   d=d, r=r, T=T_r, times=times, m=m,
                   sigma_max=args.max, sigma_min=args.min, spaces=args.spaces,
                   sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False)

    elif args.test == "r":
        test_r(study=study, opt_d=opt_d, opt_r=opt_r, opt_sigma=opt_sigma,
               d=d, sigma=sigma, T=T_r, times=times, m=m,
               r_max=args.max, r_min=args.min, spaces=args.spaces,
               sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False)

    elif args.test == "d":
        test_d(study=study, opt_d=opt_d, opt_r=opt_r, opt_sigma=opt_sigma,
               r=r, sigma=sigma, T=T_r, times=times, m=m,
               d_max=args.max, d_min=args.min, spaces=args.spaces,
               sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False)

    elif args.test == "m":
        test_m(study=study, opt_d=opt_d, opt_r=opt_r, opt_sigma=opt_sigma,
               d=d, m=m, r=r, sigma=sigma, times=times, T=T_r,
               sigma_type=matrix_type.DIAG, s_type=matrix_type.IDENTITY, load=False)

    else:
        assert False
