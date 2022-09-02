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


def create_model_delta(in_d_x, in_d_p):
    _h_delta = torch.randn(in_d_x, in_d_p) * 0.1
    c_vv_delta = torch.randn(in_d_x, in_d_x) * 0.1
    c_vv_delta = torch.diag(torch.matmul(c_vv_delta, c_vv_delta.T).diag())
    _l_delta = torch.linalg.cholesky(c_vv_delta)
    return _h_delta, _l_delta


def get_h_and_c_xx(in_model):
    return in_model.h, in_model.c_xx_bar


if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "atomic-plasma-46"

    # model, config, cnf = load_run_data(run_name)
    d_x = 16
    d_p = 2
    model = measurements_distributions.TruncatedLinearModel(d_x, d_p, 0.1, 10, a_limit=0)
    h_delta, l_delta = create_model_delta(d_x, d_p)
    h, c_xx_overline = get_h_and_c_xx(model)

    # true_crb = torch.trace(model.crb()).item() / d_p
    # opt_flow = model.get_optimal_model()
    p_true = model.parameters.random_sample_parameters()
    l_x = torch.linalg.cholesky(c_xx_overline)

    n_test = 20
    alpha_array = np.linspace(-0.5, 0.5, n_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=80)
    # plt.subplot(1, 2, 1)
    mc = pru.MetricCollector()
    for a in [-20, -10, 0]:
        mc.clear()
        model.a = a * torch.rand([d_x])
        model.b = 5
        for alpha in alpha_array:
            linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.2)

            # alpha_i = (model.a - mu) / torch.sqrt(model.c_xx_bar.diag())
            # beta_i = (model.b - mu) / model.c_xx_bar.diag()
            eps = 1e-7
            mu_overline = torch.matmul(p_true[constants.THETA], h.T)
            z_i, alpha_i, beta_i, _ = compute_truncted_normal_parameters(model.a, model.b, mu_overline,
                                                                         torch.sqrt(model.c_xx_bar.diag()))
            mu_shift = (pdf(alpha_i) - pdf(beta_i)) / z_i
            # mu_shift[z_i < eps] = 1.0
            s1 = (alpha_i * pdf(alpha_i) - beta_i * pdf(beta_i)) / z_i
            s1[z_i < eps] = 0
            s2 = torch.pow(mu_shift, 2.0)
            s2[z_i < eps] = 1
            c_xx = torch.diag(model.c_xx_bar.diag() * torch.relu(1 + s1.flatten() - s2.flatten()))
            x = model.generate_data(50000, **p_true)
            print("-" * 10)
            print(torch.var(x, dim=0))
            print(c_xx.diag())
            var_error = torch.abs(c_xx.diag() - torch.var(x, dim=0)) / (torch.var(x, dim=0) + eps)
            print(var_error)

            mcrb = linear_ms.calculate_mcrb(h, c_xx)
            mu = mu_overline + mu_shift * torch.sqrt(model.c_xx_bar.diag().reshape([1, d_x]))
            p_zero = linear_ms.calculate_pseudo_true_parameter(mu.flatten())
            lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
            gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(model.generate_data, 50000, linear_ms,
                                                                          **p_true)
            # gmcrb_cnf, gmlb_cnf = gmlb.generative_misspecified_cramer_rao_bound(cnf, 50000, linear_ms, **p_true)
            mc.insert(
                lb=torch.trace(lb).item() / d_p,
                crb=torch.trace(linear_ms.crb()) / d_p,
                mcrb=torch.trace(mcrb).item() / d_p,
                gmcrb=torch.trace(gmcrb).item() / d_p,
                # gmcrb_cnf=torch.trace(gmcrb_cnf).item() / d_p,
                # gmlb_cnf=torch.trace(gmlb_cnf).item() / d_p,
                gmlb=torch.trace(gmlb_v).item() / d_p)

        axes[0].plot(alpha_array, np.asarray(mc["gmcrb"]), "o", label=f"GMCRB (Optimal) low={a}, high=20")
        axes[0].plot(alpha_array, np.asarray(mc["mcrb"]), label=f"MCRB (Optimal) low={a}, high=20")
        axes[1].plot(alpha_array, np.asarray(mc["gmlb"]), "o", label=f"GMCRB (Optimal) low={a}, high=20")
        axes[1].plot(alpha_array, np.asarray(mc["lb"]), label=f"LB (Optimal) low={a}, high=20")
    print("a")
    axes[0].grid()
    axes[0].legend()
    axes[1].grid()
    axes[1].legend()
    plt.savefig("trunced_res.svg")
    # plt.plot(alpha_array, np.asarray(mc["gmcrb_cnf"]), "x", label="GMCRB (CNF)")
    # plt.xlabel(r"$\alpha$")
    # plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")
    # plt.plot(alpha_array, np.asarray(mc["crb"]), label="CRB (Assume)")
    # plt.legend()
    # plt.grid()
    # plt.subplot(1, 2, 2)
    # plt.plot(alpha_array, np.asarray(mc["lb"]), label="LB")
    # plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label="GMLB (Optimal)")
    # plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x", label="GMLB (CNF)")
    # plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")
    # plt.plot(alpha_array, np.asarray(mc["crb"]), label="CRB (Assume)")

    # plt.legend()
    # plt.grid()
    # plt.xlabel(r"$\alpha$")
    plt.show()

    # plt.plot(alpha_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    # plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label="LB")
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # mc.clear()
    # beta_array = np.linspace(-1.5, 1.5, 100)
    # for beta in beta_array:
    #     linear_ms = build_misspecifietion_type_one(h, c_xx, h_delta, c_vv_delta, 0.0, beta)
    #     mcrb = linear_ms.calculate_mcrb(h, c_xx)
    #     p_zero = linear_ms.calculate_pseudo_true_parameter(p_true[constants.THETA].flatten(), h)
    #     lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
    #     gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(opt_flow, 50000, linear_ms, **p_true)
    #     mc.insert(lb=torch.trace(lb).item() / d_p,
    #               mcrb=torch.trace(mcrb).item() / d_p,
    #               gmcrb=torch.trace(gmcrb).item() / d_p,
    #               gmlb=torch.trace(gmlb_v).item() / d_p)
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(beta_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    # plt.plot(beta_array, np.asarray(mc["gmcrb"]), "-o", label="GMCRB (Optimal)")
    # plt.plot(beta_array, np.ones(len(beta_array)) * true_crb, label="CRB (True)")
    # plt.legend()
    # plt.grid()
    # plt.subplot(1, 2, 2)
    # plt.plot(beta_array, np.asarray(mc["lb"]), "--", label="LB")
    # plt.plot(beta_array, np.asarray(mc["gmlb"]), "-o", label="GMLB (Optimal)")
    # plt.legend()
    #
    # plt.grid()
    # plt.show()
