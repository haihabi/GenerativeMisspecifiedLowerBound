import copy

import torch
import pyresearchutils as pru
import numpy as np
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one
from experiments import constants
import gmlb
from experiments.measurements_distributions.linear_truncated_gaussian.computing_moments import \
    compute_second_order_state
from experiments.analysis.helpers import load_run_data


def create_model_delta(in_d_x, in_d_p):
    _h_delta = torch.randn(in_d_x, in_d_p) * 0.1
    c_vv_delta = torch.randn(in_d_x, in_d_x) * 0.1
    c_vv_delta = torch.diag(torch.matmul(c_vv_delta, c_vv_delta.T).diag())
    _l_delta = torch.linalg.cholesky(c_vv_delta)
    return _h_delta, _l_delta


def get_h_and_c_xx(in_model):
    return in_model.h, in_model.c_xx_bar


def compute_mean_covarinace(in_model, in_mu_overline, eps=1e-8):
    print("a")

    lb = in_model.a.detach().cpu().numpy()
    ub = in_model.b.detach().cpu().numpy()
    c_xx_bar = in_model.c_xx_bar.detach().cpu().numpy()
    _mu, _c_xx = compute_second_order_state(lb, ub, in_mu_overline.detach().cpu().numpy().flatten(), c_xx_bar)
    # z_i, alpha_i, beta_i, _ = compute_truncted_normal_parameters(in_model.a, in_model.b, in_mu_overline,
    #                                                              torch.sqrt(in_model.c_xx_bar.diag()))
    # beta_i = beta_i.double()
    # alpha_i = alpha_i.double()
    # delta_prob = pdf(alpha_i) - pdf(beta_i)
    # zero_flag = torch.logical_and(delta_prob < eps, z_i < eps)
    # mu_shift = (pdf(alpha_i) - pdf(beta_i)) / z_i
    #
    # mu_inf_base = (in_model.a * pdf(alpha_i) - in_model.b * pdf(beta_i)) / (pdf(alpha_i) - pdf(beta_i))
    # mu_inf = mu_inf_base - in_mu_overline
    # mu_inf /= torch.sqrt(in_model.c_xx_bar.diag())
    # mu_shift[zero_flag] = mu_inf[zero_flag]
    # s1 = (alpha_i * pdf(alpha_i) - beta_i * pdf(beta_i)) / z_i
    # s1[z_i < eps] = 0
    # s2 = torch.pow(mu_shift, 2.0)
    # s2[z_i < eps] = 1
    # _c_xx = torch.diag(in_model.c_xx_bar.diag() * torch.relu(1 + s1.flatten() - s2.flatten()))
    # _mu = in_mu_overline + mu_shift * torch.sqrt(in_model.c_xx_bar.diag().reshape([1, in_mu_overline.shape[-1]]))

    return torch.tensor(_mu).float(), torch.tensor(_c_xx).float()


if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "magic-sun-145"

    model, run_parameters, cnf = load_run_data(run_name)
    h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
    h, c_xx_overline = get_h_and_c_xx(model)

    p_true = model.parameters.random_sample_parameters()
    l_x = torch.linalg.cholesky(c_xx_overline)

    n_test = 20
    alpha = 0.1
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.1)
    mc = pru.MetricCollector()
    alpha_array = np.linspace(0.1, 10, 20)
    p_true_iter = copy.copy(p_true)

    for scale in alpha_array:
        p_true_iter[constants.THETA] = p_true[constants.THETA] * scale / torch.norm(p_true[constants.THETA])

        # mu_overline = torch.matmul(p_true_iter[constants.THETA], h.T)
        s = torch.min(model.a)
        c = (model.b - model.a)[0]
        mu_overline = c * torch.arctan(3.3 * (torch.matmul(p_true_iter[constants.THETA], h.T) - s) / c) / np.pi + s

        mu, c_xx = compute_mean_covarinace(model, mu_overline)

        mcrb = linear_ms.calculate_mcrb(0, c_xx)
        p_zero = linear_ms.calculate_pseudo_true_parameter(mu.flatten())
        lb = gmlb.compute_lower_bound(mcrb, p_true_iter[constants.THETA].flatten(), p_zero)
        for i in range(1):
            gmcrb, gmlb_v, p_zero_est = gmlb.generative_misspecified_cramer_rao_bound(model.generate_data, 256000,
                                                                                      linear_ms,
                                                                                      **p_true_iter)
            gmcrb_cnf, gmlb_cnf_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 64000,
                                                                                          linear_ms,
                                                                                          min_limit=model.a,
                                                                                          max_limit=model.b,
                                                                                          **p_true_iter)
        # raise NotImplemented
        mc.insert(
            lb=torch.trace(lb).item() / run_parameters.d_p,
            crb=torch.trace(linear_ms.crb()) / run_parameters.d_p,
            mcrb=torch.trace(mcrb).item() / run_parameters.d_p,
            gmcrb=torch.trace(gmcrb).item() / run_parameters.d_p,
            gmcrb_cnf=torch.trace(gmcrb_cnf).item() / run_parameters.d_p,
            gmlb_cnf=torch.trace(gmlb_cnf_v).item() / run_parameters.d_p,
            gmlb=torch.trace(gmlb_v).item() / run_parameters.d_p)

    plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label=f"GMCRB (Optimal)")
    plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "o", label=f"GMCRB (CNF)")
    plt.plot(alpha_array, np.asarray(mc["lb"]), label=f"LB")
    # plt.plot(alpha_array, np.asarray(mc["mcrb"]), label=f"MCRB")
    # plt.plot(alpha_array, np.asarray(mc["gmcrb"]), "o", label=f"GMCRB (Optimal)")

    plt.grid()
    plt.legend()
    # axes[1].grid()
    # axes[1].legend()
    plt.tight_layout()
    plt.savefig("trunced_res.svg")
    plt.show()
