import torch
import pyresearchutils as pru
from experiments import measurements_distributions
import numpy as np
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one
from experiments import constants
import gmlb
from experiments.measurements_distributions.linear_truncated_gaussian.truncated_gaussian import \
    compute_truncted_normal_parameters, pdf
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
    z_i, alpha_i, beta_i, _ = compute_truncted_normal_parameters(in_model.a, in_model.b, in_mu_overline,
                                                                 torch.sqrt(in_model.c_xx_bar.diag()))
    beta_i = beta_i.double()
    alpha_i = alpha_i.double()
    delta_prob = pdf(alpha_i) - pdf(beta_i)
    zero_flag = torch.logical_and(delta_prob < eps, z_i < eps)
    mu_shift = (pdf(alpha_i) - pdf(beta_i)) / z_i

    mu_inf_base = (in_model.a * pdf(alpha_i) - in_model.b * pdf(beta_i)) / (pdf(alpha_i) - pdf(beta_i))
    mu_inf = mu_inf_base - in_mu_overline
    mu_inf /= torch.sqrt(in_model.c_xx_bar.diag())
    mu_shift[zero_flag] = mu_inf[zero_flag]
    s1 = (alpha_i * pdf(alpha_i) - beta_i * pdf(beta_i)) / z_i
    s1[z_i < eps] = 0
    s2 = torch.pow(mu_shift, 2.0)
    s2[z_i < eps] = 1
    _c_xx = torch.diag(in_model.c_xx_bar.diag() * torch.relu(1 + s1.flatten() - s2.flatten()))
    _mu = in_mu_overline + mu_shift * torch.sqrt(in_model.c_xx_bar.diag().reshape([1, in_mu_overline.shape[-1]]))

    return _mu.float(), _c_xx.float()


if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "spring-totem-91"

    model, run_parameters, cnf = load_run_data(run_name)
    h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
    h, c_xx_overline = get_h_and_c_xx(model)

    p_true = model.parameters.random_sample_parameters()
    l_x = torch.linalg.cholesky(c_xx_overline)

    n_test = 20
    alpha_array = np.linspace(-0.5, 0.5, n_test)
    # fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=80)
    mc = pru.MetricCollector()

    for alpha in alpha_array:
        linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.0)

        mu_overline = torch.matmul(p_true[constants.THETA], h.T)
        mu, c_xx = compute_mean_covarinace(model, mu_overline)

        mcrb = linear_ms.calculate_mcrb(h, c_xx)
        p_zero = linear_ms.calculate_pseudo_true_parameter(mu.flatten())
        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v, p_zero_est = gmlb.generative_misspecified_cramer_rao_bound(model.generate_data, 128000,
                                                                                  linear_ms,
                                                                                  **p_true)

        gmcrb_cnf, _, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 128000,
                                                                             linear_ms,
                                                                             # min_limit=model.a,
                                                                             # max_limit=model.b,
                                                                             **p_true)
        mc.insert(
            lb=torch.trace(lb).item() / run_parameters.d_p,
            crb=torch.trace(linear_ms.crb()) / run_parameters.d_p,
            mcrb=torch.trace(mcrb).item() / run_parameters.d_p,
            gmcrb=torch.trace(gmcrb).item() / run_parameters.d_p,
            gmcrb_cnf=torch.trace(gmcrb_cnf).item() / run_parameters.d_p,
            gmlb=torch.trace(gmlb_v).item() / run_parameters.d_p)

    plt.plot(alpha_array, np.asarray(mc["gmcrb"]), "o", label=f"GMCRB (Optimal)")
    plt.plot(alpha_array, np.asarray(mc["gmcrb_cnf"]), "o", label=f"GMCRB (CNF)")
    plt.plot(alpha_array, np.asarray(mc["mcrb"]), label=f"MCRB")
    # axes[1].plot(alpha_array, np.asarray(mc["gmlb"]), "o", label=f"GMCRB (Optimal) low={a}, high=20")
    # axes[1].plot(alpha_array, np.asarray(mc["lb"]), label=f"LB (Optimal) low={a}, high=20")
    plt.grid()
    plt.legend()
    # axes[1].grid()
    # axes[1].legend()
    plt.tight_layout()
    plt.savefig("trunced_res.svg")
    plt.show()
