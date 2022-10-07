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
from experiments.analysis.helpers import load_run_data, create_model_delta
from experiments.measurements_distributions.linear_truncated_gaussian.softclip import soft_clip


def get_h_and_c_xx(in_model):
    return in_model.h, in_model.c_xx_bar


def compute_mean_covarinace(in_model, in_mu_overline):
    lb = in_model.a.detach().cpu().numpy()
    ub = in_model.b.detach().cpu().numpy()
    c_xx_bar = in_model.c_xx_bar.detach().cpu().numpy()
    _mu, _c_xx = compute_second_order_state(lb, ub, in_mu_overline.detach().cpu().numpy().flatten(), c_xx_bar)
    return torch.tensor(_mu).float(), torch.tensor(_c_xx).float()


if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "glad-smoke-154"
    alpha = 0.1
    beta = 0.1
    n_test = 20

    model, run_parameters, cnf = load_run_data(run_name)
    h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
    h, c_xx_overline = get_h_and_c_xx(model)
    l_x = torch.linalg.cholesky(c_xx_overline)
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
    p_true = model.parameters.random_sample_parameters()

    mc = pru.MetricCollector()
    alpha_array = np.linspace(0.1, 10, 20)
    p_true_iter = copy.copy(p_true)

    for scale in alpha_array:
        p_true_iter[constants.THETA] = p_true[constants.THETA] * scale / torch.norm(p_true[constants.THETA])

        mu_overline = soft_clip(torch.matmul(p_true_iter[constants.THETA], h.T), torch.min(model.a), torch.max(model.b))
        # s = torch.min(model.a)
        # c = (model.b - model.a)[0]
        # mu_overline = c * torch.arctan(3.3 * (torch.matmul(p_true_iter[constants.THETA], h.T) - s) / c) / np.pi + s

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
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("trunced_res.svg")
    plt.show()
