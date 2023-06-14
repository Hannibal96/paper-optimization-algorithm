from lr_steps import *
import matplotlib.pyplot as plt
import pickle
import time
from lr_utils import plot_and_save
import argparse


def run_lr_algo(d, r, b, sigma,
                lr_r, lr_f, beta_r, beta_f,
                m, avg_frac, stop_frac, T_r, T_f,
                times):
    x = np.random.randn(b, d, 1) * sigma
    regrets = np.zeros([times, m+1])
    for t in range(times):
        start_time = time.time()
        print(f"#Exp={t}")
        R, f, p, o, regrets_t = algorithm(x=x, r=r,
                                       lr_r=lr_r, lr_f=lr_f, beta1=beta_r, beta2=beta_f,
                                       m=m, T_r=T_r, T_f=T_f, avg_frac=avg_frac, stop_frac=stop_frac)
        regrets[t, :] = regrets_t
        print(regrets_t[-1])
        run_time_seconds = time.time() - start_time
        print(f"Running Time: {run_time_seconds // 60}:{run_time_seconds % 60}")

    return regrets


def test_m(d, r, b, sigma, lr_r, lr_f, beta_r, beta_f, m, avg_frac, stop_frac, T_r, T_f, times):
    regrets = run_lr_algo(d=d, r=r, b=b, sigma=sigma,
                          lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f,
                          m=m, avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f,
                          times=times)
    plt.grid()
    plt.xlabel('#Iteration')
    plt.ylabel('Regret')
    plt.boxplot(regrets)

    name = f"log-reg_reg-vs-iter_d={d}_r={r}_b={b}"
    with open(f'{name}.p', 'wb') as f:
        pickle.dump(regrets, f)
    plt.savefig(f"{name}.png")
    plt.clf()


def test_d(r, b, sigma, lr_r, lr_f, beta_r, beta_f, avg_frac, stop_frac, T_r, T_f, times,
           d_start, d_max, d_step, max_m, m_step):

    m_list = list(range(1, max_m + 1, m_step))
    d_list = list(range(d_start, d_max + 1, d_step))

    mean_regrets = np.zeros([len(m_list), len(d_list)])
    std_regrets = np.zeros([len(m_list), len(d_list)])
    regrets_tensor = np.zeros([len(m_list), len(d_list), times])

    for m_idx, m in enumerate(m_list):
        for d_idx, d in enumerate(d_list):
            print(f"*** m={m}, d={d} ***")
            regrets = run_lr_algo(d=d, r=r, b=b, sigma=sigma,
                                  lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f,
                                  m=m, avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f,
                                  times=times)

            regrets_tensor[m_idx, d_idx] = regrets.min(axis=1)
            mean_regrets[m_idx, d_idx] = regrets.min(axis=1).mean()
            std_regrets[m_idx, d_idx] = regrets.min(axis=1).std()

    plot_and_save(x_axis=d_list, mean_regrets=mean_regrets, std_regrets=std_regrets,
                  label_char=r"$m$", labels_val=m_list, x_label=r'$d$', y_label=f'Regret',
                  title=r"logistic regression regret vs $d$" + f"_r={r}",
                  name=f"reg-vs-d_r={r}_b={b}")


def test_sigma(d, r, b, lr_r, lr_f, beta_r, beta_f, avg_frac, stop_frac, T_r, T_f, times,
               s_start, s_max, s_step, max_m, m_step):

    m_list = list(range(1, max_m + 1, m_step))
    s_list = np.linspace(s_start, s_max, int((s_max - s_start) / s_step) + 1)

    mean_regrets = np.zeros([len(m_list), len(s_list)])
    std_regrets = np.zeros([len(m_list), len(s_list)])
    for m_idx, m in enumerate(m_list):
        for s_idx, sigma in enumerate(s_list):
            print(f"*** m={m}, sigma={sigma} ***")
            regrets = run_lr_algo(d=d, r=r, b=b, sigma=sigma,
                                  lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f,
                                  m=m, avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f,
                                  times=times)
            mean_regrets[m_idx, s_idx] = regrets.min(axis=1).mean()
            std_regrets[m_idx, s_idx] = regrets.min(axis=1).std()

    plot_and_save(x_axis=s_list, mean_regrets=mean_regrets, std_regrets=std_regrets,
                  label_char=r"$m$", labels_val=m_list, x_label=r'$\sigma_0^2$', y_label=f'Regret',
                  title=r"logistic regression regret vs $\sigma_0^2$" + f"_r={r}",
                  name=f"reg-vs-sigma_r={r}_b={b}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Logistic Regression setting")
    parser.add_argument('--b', type=int, help="The batch size of the data", required=True)
    parser.add_argument('--d', type=int, help="The d value to test", required=True)
    parser.add_argument('--r', type=int, help="The r value to test", required=True)
    parser.add_argument('--sigma', "--s", type=int, help="The sigma value to test", required=True)
    parser.add_argument('--m', type=int, help="The number of iterations of the algorithm", required=True)

    parser.add_argument('--times', type=int, help="The number of times to run the algorithm", required=True)
    parser.add_argument('--Tr', '--tr', type=int, help="The number of steps of the finding R phase", required=True)
    parser.add_argument('--Tf', '--tf', type=int, help="The number of steps of the finding F phase", required=True)

    parser.add_argument('--lr_r', type=float, help="Learning rate R phase", required=True)
    parser.add_argument('--lr_f', type=float, help="Learning rate F phase", required=True)
    parser.add_argument('--beta_r', type=float, help="Beta for R phase", required=True)
    parser.add_argument('--beta_f', type=float, help="Beta for F phase", required=True)
    parser.add_argument('--avg_frac', type=int, help="Average fraction", required=True)
    parser.add_argument('--stop_frac', type=int, help="Stop fraction", required=True)

    parser.add_argument('--test', '--Test', type=str, help="Parameter test", required=True, choice=["sigma", "m", "d"])

    parser.add_argument('--min', type=int, help="Min value to test", required=False)
    parser.add_argument('--max', type=int, help="Max value to test", required=False)
    parser.add_argument('--step', type=int, help="Step value to test", required=False)
    parser.add_argument('--m_max', type=int, help="m max value to test", required=False)
    parser.add_argument('--m_step', type=int, help="m step value to test", required=False)

    args = parser.parse_args()

    b = args.b
    d = args.d
    r = args.r
    sigma = args.sigma
    m = args.m
    times = args.times

    lr_r = args.lr_r
    lr_f = args.lr_f
    beta_r = args.beta_r
    beta_f = args.beta_f

    T_r = args.Tr
    T_f = args.Tf

    avg_frac = args.avg_frac
    stop_frac = args.stop_frac

    if args.test == "m":
        test_m(d=d, r=r, b=b, sigma=sigma, lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f, m=m,
               avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f, times=times)

    if args.test == "d":
        test_d(r=r, b=b, sigma=sigma, lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f,
               avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f, times=times,
               d_start=args.min, d_max=args.max, d_step=args.step, max_m=args.m_max, m_step=args.m_step)

    if args.test == "sigma":
        test_sigma(d=d, r=r, b=b, lr_r=lr_r, lr_f=lr_f, beta_r=beta_r, beta_f=beta_f,
                   avg_frac=avg_frac, stop_frac=stop_frac, T_r=T_r, T_f=T_f, times=times,
                   s_start=args.min, s_max=args.max, s_step=args.step, max_m=args.m_max, m_step=args.m_step)

