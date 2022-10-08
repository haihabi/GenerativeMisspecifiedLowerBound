import copy

import torch
import pyresearchutils as pru
import numpy as np
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one, parameter_sweep
from experiments import constants
import gmlb
from experiments.measurements_distributions.linear_truncated_gaussian.computing_moments import \
    compute_second_order_state
from experiments.analysis.helpers import load_run_data, create_model_delta, get_h_and_c_xx
from experiments.measurements_distributions.linear_truncated_gaussian.softclip import soft_clip

if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "earnest-disco-159"
    alpha = 0.1
    beta = 0.1
    n_test = 20
    d_p = 8
    generate_delta = True
    norm_max = True
    m = 640000
    alpha_array = np.linspace(0.1, 10, 20)
    for run_name in ["treasured-surf-177", "bright-aardvark-180"]:
        model, run_parameters, cnf = load_run_data(run_name)
        if generate_delta:
            generate_delta = False
            h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
            h, c_xx_overline = get_h_and_c_xx(model)
            l_x = torch.linalg.cholesky(c_xx_overline)
            linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
            p_true = model.parameters.random_sample_parameters()

        gmcrb_est_array, mcrb_est_array, lb_array_z, norm_array = parameter_sweep(cnf, p_true, n_test, linear_ms, m,
                                                                                  model,
                                                                                  norm_max=norm_max, run_optimal=True,
                                                                                  non_linear=run_parameters.non_linear_function)

        plt.plot(norm_array.detach().numpy(),
                 (torch.diagonal(lb_array_z, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(),
                 label=f"LB")
        plt.plot(norm_array.detach().numpy(),
                 (torch.diagonal(mcrb_est_array, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(), "o",
                 label=f"GMLB (Optimal)")
        plt.plot(norm_array.detach().numpy(),
                 (torch.diagonal(gmcrb_est_array, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(), "x",
                 label=f"GMLB (Trained)")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\frac{\mathrm{Tr}(LB)}{d_p}$")
    plt.tight_layout()
    plt.show()
    # print("---")
    # print(h_delta)

    # mc = pru.MetricCollector()
    # p_true_iter = copy.copy(p_true)
    # for scale in alpha_array:
    #     p_true_iter[constants.THETA] = p_true[constants.THETA] * scale / torch.norm(p_true[constants.THETA])
    #
    #     # mu_overline = soft_clip(torch.matmul(p_true_iter[constants.THETA], h.T), torch.min(model.a),
    #     #                         torch.max(model.b))
    #     mu_overline = torch.matmul(p_true_iter[constants.THETA], h.T)
    #
    #     mu, c_xx = compute_mean_covarinace(model, mu_overline)
    #
    #     mcrb = linear_ms.calculate_mcrb(0, c_xx)
    #     p_zero = linear_ms.calculate_pseudo_true_parameter(mu.flatten())
    #     lb = gmlb.compute_lower_bound(mcrb, p_true_iter[constants.THETA].flatten(), p_zero)
    #
    #     gmcrb, gmlb_v, p_zero_est = gmlb.generative_misspecified_cramer_rao_bound(model.generate_data, 256000,
    #                                                                               linear_ms,
    #                                                                               **p_true_iter)
    #     gmcrb_cnf, gmlb_cnf_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 256000,
    #                                                                                   linear_ms,
    #                                                                                   min_limit=model.a,
    #                                                                                   max_limit=model.b,
    #                                                                                   **p_true_iter)
    #
    #     mc.insert(
    #         lb=torch.trace(lb).item() / run_parameters.d_p,
    #         crb=torch.trace(linear_ms.crb()) / run_parameters.d_p,
    #         mcrb=torch.trace(mcrb).item() / run_parameters.d_p,
    #         gmcrb=torch.trace(gmcrb).item() / run_parameters.d_p,
    #         gmcrb_cnf=torch.trace(gmcrb_cnf).item() / run_parameters.d_p,
    #         gmlb_cnf=torch.trace(gmlb_cnf_v).item() / run_parameters.d_p,
    #         gmlb=torch.trace(gmlb_v).item() / run_parameters.d_p)

    plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label=f"GMLB (Optimal) $a=$" + f"{run_parameters.min_limit}")
    plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x",
             label=f"GMLB (Trained) $a=$" + f"{run_parameters.min_limit}")
    plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label=f"LB $a=$" + f"{run_parameters.min_limit}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("trunced_res.svg")
plt.show()
