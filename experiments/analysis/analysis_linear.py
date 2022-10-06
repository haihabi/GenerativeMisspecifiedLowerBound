import copy

import numpy as np
import torch

from experiments import constants
import gmlb
from matplotlib import pyplot as plt
import pyresearchutils as pru
from experiments.analysis.helpers import build_misspecifietion_type_one, load_run_data


def get_h_and_c_xx(in_opt_flow):
    _h = in_opt_flow.flow.flows[0].h
    _c_xx = in_opt_flow.flow.flows[0].c_xx
    return _h, _c_xx


def parameter_sweep(in_flow, in_p_true, in_n_test_points, in_linear_ms, in_samples_per_point):
    norm_array = torch.linspace(0.1, 3, in_n_test_points)
    res_list = []
    for norm in norm_array:
        p_true[constants.THETA] = in_p_true[constants.THETA] / torch.norm(in_p_true[constants.THETA])
        p_true[constants.THETA] = in_p_true[constants.THETA] * norm
        _, gmlb_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(in_flow,
                                                                          in_samples_per_point,
                                                                          in_linear_ms, **p_true)
        res_list.append(gmlb_v)
    return torch.stack(res_list)


DATASET_SIZE2RUNNAME = {2: {200: "earthy-field-1051",
                            2000: "prime-fog-103",
                            20000: "dark-thunder-104",
                            200000: "solar-plasma-102"},
                        4: {200: "earthy-fog-117",
                            2000: "different-terrain-112",
                            20000: "fresh-haze-110",
                            200000: "wise-aardvark-125"},
                        8: {200: "breezy-leaf-131",
                            2000: "northern-universe-129",
                            20000: "fragrant-sun-127",
                            200000: "charmed-resonance-122"}
                        }

if __name__ == '__main__':
    pru.set_seed(0)
    # run_name = "woven-snow-50"
    run_name = "charmed-resonance-122"
    model, config, cnf = load_run_data(run_name)
    dataset_size = config.dataset_size
    d_x = config.d_x
    d_p = config.d_p

    h_delta = torch.randn(d_x, d_p) * 0.1
    c_vv_delta = torch.randn(d_x, d_x) * 0.1
    c_vv_delta = torch.matmul(c_vv_delta, c_vv_delta.T)

    true_crb = torch.trace(model.crb()).item() / d_p
    opt_flow = model.get_optimal_model()
    p_true = model.parameters.random_sample_parameters()
    h, c_xx = get_h_and_c_xx(opt_flow)
    l_x = torch.linalg.cholesky(c_xx)
    l_delta = torch.linalg.cholesky(c_vv_delta)
    n_test = 20

    mc = pru.MetricCollector()
    alpha = 0.1
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.1)
    mc.clear()
    p_true_iter = copy.copy(p_true)
    alpha_array = np.linspace(0.1, 10, 20)
    mcrb = linear_ms.calculate_mcrb(h, c_xx)
    for scale in alpha_array:
        p_true_iter[constants.THETA] = p_true[constants.THETA] * scale / torch.norm(p_true[constants.THETA])

        mu = torch.matmul(h, p_true_iter[constants.THETA].flatten())
        p_zero = linear_ms.calculate_pseudo_true_parameter(mu)

        lb = gmlb.compute_lower_bound(mcrb, p_true_iter[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(opt_flow, 50000, linear_ms, **p_true_iter)
        gmcrb_cnf, gmlb_cnf, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 50000, linear_ms,
                                                                                    **p_true_iter)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  crb=torch.trace(linear_ms.crb()) / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmcrb_cnf=torch.trace(gmcrb_cnf).item() / d_p,
                  gmlb_cnf=torch.trace(gmlb_cnf).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(alpha_array, np.asarray(mc["lb"]), label="LB")
    plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label="GMLB (Optimal)")
    plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x", label="GMLB (CNF)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlabel(r"$\alpha$")
    plt.savefig("compare.svg")
    plt.show()

    p_true = model.parameters.random_sample_parameters()

    n_test_points = 20
    dataset_array = [200, 2000, 20000, 200000]
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, 0.1, 0.0)
    mcrb = linear_ms.calculate_mcrb(h, c_xx)
    spp_array = [32, 64, 128, 256, 512, 1024, 2048]

    mean_re_list = []
    max_re_list = []
    for sample_per_point in spp_array:
        gmcrb_est_array = parameter_sweep(opt_flow, p_true, n_test_points, linear_ms, sample_per_point)
        gre = torch.norm(gmcrb_est_array - torch.unsqueeze(mcrb, dim=0), dim=(1, 2)) / torch.norm(mcrb)
        mean_re_list.append(torch.mean(gre).item())
        max_re_list.append(torch.max(gre).item())

    plt.semilogy(spp_array, mean_re_list, label=r"Mean")
    plt.semilogy(spp_array, max_re_list, label=r"Max")
    plt.legend()
    plt.xlabel("N-Samples[k]")
    plt.ylabel("xRE")
    plt.grid()
    plt.tight_layout()
    plt.show()

    mean_re_list = []
    gmean_re_list = []
    max_re_list = []
    for dataset_size in dataset_array:
        run_name = DATASET_SIZE2RUNNAME[8][dataset_size]
        _, _, cnf = load_run_data(run_name)
        samples_per_point = int(dataset_size / n_test_points)
        mcrb_est_array = parameter_sweep(opt_flow, p_true, n_test_points, linear_ms, samples_per_point)
        gmcrb_est_array = parameter_sweep(cnf, p_true, n_test_points, linear_ms, 128000)
        re = torch.norm(mcrb_est_array - torch.unsqueeze(mcrb, dim=0), dim=(1, 2)) / torch.norm(mcrb)
        gre = torch.norm(gmcrb_est_array - torch.unsqueeze(mcrb, dim=0), dim=(1, 2)) / torch.norm(mcrb)
        mean_re_list.append(torch.mean(re).item())
        gmean_re_list.append(torch.mean(gre).item())
        max_re_list.append(torch.max(re).item())
    plt.semilogx(dataset_array, mean_re_list, label=r"$\overline{\mathrm{LB}}$")
    plt.semilogx(dataset_array, gmean_re_list, label=r"$\mathrm{GMLB}$")
    plt.legend()
    plt.xlabel("Dataset-size")
    plt.ylabel("xRE")
    plt.grid()
    plt.tight_layout()
    plt.savefig("dataset-size-effect.svg")
    plt.show()
