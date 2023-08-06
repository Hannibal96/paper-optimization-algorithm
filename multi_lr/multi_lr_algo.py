from multi_lr_sol import *


def find_new_f(full_labels, data, R, p):
    worst = 0
    worst_f = None

    for f_idx, labels in enumerate(full_labels.T[1:]):
        total_loss = 0
        for curr_R, curr_p in zip(R, p):
            w = find_w(R=curr_R, x=data, y=labels)
            loss = calc_loss(x=x, R=R, w=w, labels=labels) * curr_p
            total_loss += curr_p * loss
        if total_loss > worst:
            worst = total_loss
            worst_f = f_idx
    return shape_idx[worst_f+1]


def find_new_R():
    R = np.random.randn(d, r)
    for i in range(4):
        w = find_w(R=R, x=x, y=labels, lr=w_lr, tol=tol)
        R = find_R(w=w, x=x, y=labels, r=r, lr=r_lr, T=T)


def mu_lr_algo(data, full_labels):
    R = np.random.randn(d, 1) * 0
    R = [R]
    new_f = find_new_f(full_labels=full_labels, data=data, R=R, p=[1])
    print(new_f)


if __name__ == "__main__":
    data, full_labels = gen_data(N=1000)
    x = data.reshape(data.shape[0], -1)
    d = x.shape[1]
    mu_lr_algo(data=x, full_labels=full_labels)


