import numpy as np
from scipy.stats import norm
from scipy.stats import mvn


def take_ind(in_matrix, in_ind):
    m = len(in_ind)
    out_mat = np.zeros([m, m])
    for i, i_tilde in enumerate(in_ind):
        for j, j_tilde in enumerate(in_ind):
            out_mat[i, j] = in_matrix[i_tilde, j_tilde]
    return out_mat


def compute_f_zero(in_lb, in_ub, in_mu, in_cov):
    return mvn.mvnun(in_lb, in_ub, in_mu, in_cov)[0]


def build_sub_vector_moments(index, in_n, in_lb, in_ub, in_mu, in_cov):
    ind2 = np.linspace(0, in_n - 1, in_n).astype("int")
    ind2 = np.asarray([j for j in ind2 if j != index])
    ai = in_lb[ind2]
    bi = in_ub[ind2]
    mui = in_mu[ind2]
    s_i = in_cov[ind2, index]
    ss_i = take_ind(in_cov, ind2) - np.matmul(s_i.reshape([-1, 1]), s_i.reshape([-1, 1]).T) / in_cov[index, index]
    mai = mui + (in_lb[index] - in_mu[index]) * s_i / in_cov[index, index]
    mbi = mui + (in_ub[index] - in_mu[index]) * s_i / in_cov[index, index]
    return ai, bi, mai, mbi, ss_i


def compute_f_1(in_lb, in_ub, in_mu, in_cov, e_index=None):
    n = len(in_lb)
    f_0 = compute_f_zero(in_lb, in_ub, in_mu, in_cov)
    s = np.sqrt(np.diag(in_cov))
    pdfa = norm.pdf(in_lb, in_mu, s)
    pdfb = norm.pdf(in_ub, in_mu, s)
    c_list = []
    for i in range(n):
        ai, bi, mai, mbi, ss_i = build_sub_vector_moments(i, n, in_lb, in_ub, in_mu, in_cov)

        f_0_lb = compute_f_zero(ai, bi, mai, ss_i)
        f_0_ub = compute_f_zero(ai, bi, mbi, ss_i)
        c_list.append(pdfa[i] * f_0_lb - pdfb[i] * f_0_ub)
    c = np.asarray(c_list)
    if e_index is None:
        return in_mu * f_0 + np.matmul(in_cov, c), f_0
    else:
        return (in_mu * f_0 + np.matmul(in_cov, c))[e_index]


def compute_second_order_state(in_lb, in_ub, in_mu, in_cov):
    n = len(in_lb)
    s = np.sqrt(np.diag(in_cov))
    pdfa = norm.pdf(in_lb, in_mu, s)
    pdfb = norm.pdf(in_ub, in_mu, s)
    f_1, f_0 = compute_f_1(in_lb, in_ub, in_mu, in_cov)
    mu_vector = f_1 / f_0

    cov_matrix = in_mu.reshape([-1, 1]) * f_1.reshape([1, -1])
    for j in range(n):
        ##########################################
        # Build Ck vector
        ##########################################
        c_vector = []
        for m in range(n):
            ai, bi, mai, mbi, ss_i = build_sub_vector_moments(m, n, in_lb, in_ub, in_mu, in_cov)
            if m == j:
                f_0_lb = compute_f_zero(ai, bi, mai, ss_i)
                f_0_ub = compute_f_zero(ai, bi, mbi, ss_i)
                c_j = f_0 + in_lb[m] * pdfa[m] * f_0_lb - in_ub[m] * pdfb[m] * f_0_ub
            else:
                e_index = j
                if j > m:
                    e_index = j - 1
                f_1_lb = compute_f_1(ai, bi, mai, ss_i, e_index=e_index)
                f_1_ub = compute_f_1(ai, bi, mbi, ss_i, e_index=e_index)
                c_j = pdfa[m] * f_1_lb - pdfb[m] * f_1_ub
            c_vector.append(c_j)
        s = np.matmul(in_cov, np.asarray(c_vector))
        cov_matrix[j, :] += s
    cov_matrix = cov_matrix / f_0 - np.matmul(mu_vector.reshape([-1, 1]), mu_vector.reshape([-1, 1]).T)

    return mu_vector, cov_matrix

# if __name__ == '__main__':
#     import pyresearchutils as pru
#     from experiments.measurements_distributions.linear_truncated_gaussian.minimax_tilting_sampler import TruncatedMVN
#
#     n = 8
#     a = 0
#     b = 2.0
#
#     mu = np.zeros(n) + 0.2  # + 2 * np.random.rand(n)
#     L = np.random.randn(n, n)
#     c_xx = np.diag(np.ones(n))
#     c_xx[1, 0] = c_xx[0, 1] = 0.8
#
#     n_samples_test = 100000
#     samples_test = TruncatedMVN(mu, np.copy(c_xx), np.ones(n) * a, np.ones(n) * b).sample(n_samples_test)
#     mu_trunc = samples_test.mean(axis=-1)
#     dv = np.expand_dims(samples_test.T, axis=-1)
#     cov_trunc = np.matmul(dv, np.transpose(dv, axes=(0, 2, 1))).mean(axis=0)
#     pru.tic()
#     mu, conv = compute_second_order_state(np.ones(n) * a, np.ones(n) * b, mu, np.copy(c_xx))
#     pru.toc()
#     print(mu_trunc)
#     print(mu)
#     pass
