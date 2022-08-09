import numpy as np
import torch

from experiments import constants
from experiments.mcrb.linear_mcrb import LinearMCRB
from experiments import measurements_distributions
import gmlb
from matplotlib import pyplot as plt
import pyresearchutils as pru


def get_h_and_c_xx(in_opt_flow):
    _h = in_opt_flow.flow.flows[0].h
    _c_xx = in_opt_flow.flow.flows[0].c_xx
    return _h, _c_xx


def build_misspecifietion_type_one(in_h, in_l_xx, in_h_delta, in_l_vv_delta, in_alpha, in_beta):
    h_matrix = in_h + in_alpha * in_h_delta
    l_c = in_l_xx + in_beta * in_l_vv_delta
    c_ww_matrix = torch.matmul(l_c, l_c.T)
    return LinearMCRB(h_matrix, c_ww_matrix)


if __name__ == '__main__':
    pru.set_seed(0)
    d_x = 10
    d_p = 2
    norm_min = 0.1
    norm_max = 10
    h_delta = torch.randn(d_x, d_p) * 0.1
    c_vv_delta = torch.randn(d_x, d_x) * 0.1
    c_vv_delta = torch.matmul(c_vv_delta, c_vv_delta.T)
    model = measurements_distributions.LinearModel(d_x, d_p, norm_min, norm_max)
    true_crb = torch.trace(model.crb()).item() / d_p
    opt_flow = model.get_optimal_model()
    p_true = model.parameters.random_sample_parameters()
    h, c_xx = get_h_and_c_xx(opt_flow)
    l_x = torch.linalg.cholesky(c_xx)
    l_delta = torch.linalg.cholesky(c_vv_delta)
    alpha_array = np.linspace(-1.5, 1.5, 100)

    mc = pru.MetricCollector()

    mc.clear()
    for alpha in alpha_array:
        linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.0)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)
        p_zero = linear_ms.calculate_pseudo_true_parameter(p_true[constants.THETA].flatten(), h)
        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(opt_flow, 50000, linear_ms, **p_true)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)

    plt.subplot(1, 2, 1)
    plt.plot(alpha_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    plt.plot(alpha_array, np.asarray(mc["gmcrb"]), "-o", label="GMCRB (Optimal)")
    plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label="LB")
    plt.plot(alpha_array, np.asarray(mc["gmlb"]), "-o", label="GMLB (Optimal)")
    plt.legend()

    plt.grid()
    plt.show()

    mc.clear()
    beta_array = np.linspace(-1.5, 1.5, 100)
    for beta in beta_array:
        linear_ms = build_misspecifietion_type_one(h, c_xx, h_delta, c_vv_delta, 0.0, beta)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)
        p_zero = linear_ms.calculate_pseudo_true_parameter(p_true[constants.THETA].flatten(), h)
        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(opt_flow, 50000, linear_ms, **p_true)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)

    plt.subplot(1, 2, 1)
    plt.plot(beta_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    plt.plot(beta_array, np.asarray(mc["gmcrb"]), "-o", label="GMCRB (Optimal)")
    plt.plot(beta_array, np.ones(len(beta_array)) * true_crb, label="CRB (True)")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(beta_array, np.asarray(mc["lb"]), "--", label="LB")
    plt.plot(beta_array, np.asarray(mc["gmlb"]), "-o", label="GMLB (Optimal)")
    plt.legend()

    plt.grid()
    plt.show()
