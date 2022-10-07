import copy

import numpy as np
import torch

from experiments import constants
import gmlb
from matplotlib import pyplot as plt
import pyresearchutils as pru
from experiments.analysis.helpers import build_misspecifietion_type_one, load_run_data, create_model_delta, \
    parameter_sweep


def get_h_and_c_xx(in_opt_flow):
    _h = in_opt_flow.flow.flows[0].h
    _c_xx = in_opt_flow.flow.flows[0].c_xx
    return _h, _c_xx


MAX_DATASET_SIZE = 200000
DATASET_SIZE2RUNNAME = {2: {200: "earthy-field-105",
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
    run_name = "charmed-resonance-122"
    alpha = 0.1
    beta = 0.1
    n_test = 20

    model, config, cnf = load_run_data(run_name)
    h_delta, l_delta = create_model_delta(config.d_x, config.d_p)
    h, c_xx = get_h_and_c_xx(model.get_optimal_model())
    l_x = torch.linalg.cholesky(c_xx)
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
    p_true = model.parameters.random_sample_parameters()

    dataset_size = config.dataset_size
    d_x = config.d_x
    d_p = config.d_p

    opt_flow = model.get_optimal_model()

    mc = pru.MetricCollector()

    mc.clear()
    p_true_iter = copy.copy(p_true)
    alpha_array = np.linspace(0.1, 10, n_test)
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
    plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x", label="GMLB (Trained)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlabel(r"$\alpha$")
    plt.savefig("compare.svg")
    plt.show()
    raise NotImplemented
    # p_true = model.parameters.random_sample_parameters()

    # dataset_array = [200, 2000, 20000, 200000]
    # linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, 0.1, 0.0)
    # mcrb = linear_ms.calculate_mcrb(h, c_xx)
    # spp_array = [32, 64, 128, 256, 512, 1024, 2048]

    # mean_re_list = []
    # max_re_list = []
    # for sample_per_point in spp_array:
    #     gmcrb_est_array = parameter_sweep(opt_flow, p_true, n_test_points, linear_ms, sample_per_point)
    #     gre = torch.norm(gmcrb_est_array - torch.unsqueeze(mcrb, dim=0), dim=(1, 2)) / torch.norm(mcrb)
    #     mean_re_list.append(torch.mean(gre).item())
    #     max_re_list.append(torch.max(gre).item())
    #
    # plt.semilogy(spp_array, mean_re_list, label=r"Mean")
    # plt.semilogy(spp_array, max_re_list, label=r"Max")
    # plt.legend()
    # plt.xlabel("N-Samples[k]")
    # plt.ylabel("xRE")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    results_p_size = {}
    for p in DATASET_SIZE2RUNNAME.keys():
        mean_re_list = []
        gmean_re_list = []
        max_re_list = []
        gmax_re_list = []

        run_name = DATASET_SIZE2RUNNAME[p][MAX_DATASET_SIZE]
        model, config, _ = load_run_data(run_name)
        p_true = model.parameters.random_sample_parameters()
        opt_flow = model.get_optimal_model()

        h_delta, l_delta = create_model_delta(config.d_x, config.d_p)
        h, c_xx = get_h_and_c_xx(model.get_optimal_model())
        l_x = torch.linalg.cholesky(c_xx)
        linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)

        # mcrb_est_array, lb_array_z = parameter_sweep(opt_flow, p_true, n_test_points, linear_ms, 128000, mcrb,
        #                                              h)

        for dataset_size in DATASET_SIZE2RUNNAME[p].keys():
            samples_per_point = int(dataset_size / n_test)

            run_name = DATASET_SIZE2RUNNAME[p][dataset_size]
            _, _, cnf = load_run_data(run_name)

            mcrb_est_array, lb_array_z = parameter_sweep(opt_flow, p_true, n_test, linear_ms, samples_per_point,
                                                         mcrb,
                                                         h)
            gmcrb_est_array, lb_array_o = parameter_sweep(cnf, p_true, n_test, linear_ms, 128000, mcrb,
                                                          h)
            re = torch.norm(mcrb_est_array - lb_array_z, dim=(1, 2)) / torch.norm(lb_array_z, dim=(1, 2))
            gre = torch.norm(gmcrb_est_array - lb_array_z, dim=(1, 2)) / torch.norm(lb_array_z, dim=(1, 2))
            mean_re_list.append(torch.mean(re).item())
            gmean_re_list.append(torch.mean(gre).item())
            max_re_list.append(torch.max(re).item())
            gmax_re_list.append(torch.max(gre).item())
        results_p_size.update({p: [mean_re_list, gmean_re_list, max_re_list, gmax_re_list]})
    for p, r in results_p_size.items():
        if p != 4:
            plt.semilogx(DATASET_SIZE2RUNNAME[p].keys(), r[0], "--", label=r"$\overline{\mathrm{LB}}$ $d_p$=" + f"{p}")
            plt.semilogx(DATASET_SIZE2RUNNAME[p].keys(), r[1], label=r"$\mathrm{GMLB}$ $d_p$=" + f"{p}")
    plt.legend()
    plt.xlabel("Dataset-size")
    plt.ylabel("xRE")
    plt.grid()
    plt.tight_layout()
    plt.savefig("dataset-size-effect.svg")
    plt.show()
