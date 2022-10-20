import copy

import numpy as np
import torch

from experiments import constants, measurements_distributions
import gmlb
from matplotlib import pyplot as plt
import pyresearchutils as pru
from experiments.analysis.helpers import build_misspecifietion_type_one, load_run_data, create_model_delta, \
    parameter_sweep, get_h_and_c_xx
from tqdm import tqdm

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
    alpha = 1.0
    beta = 1.0
    m = 64000
    norm_min = 0.2
    norm_max = 9
    n_test = 20
    model, config, cnf = load_run_data(run_name)
    h_delta, l_delta = create_model_delta(config.d_x, config.d_p)
    h, c_xx = get_h_and_c_xx(model)
    l_x = torch.linalg.cholesky(c_xx)
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
    p_true = model.parameters.random_sample_parameters()
    p_true_iter = copy.copy(p_true)
    dataset_size = config.dataset_size
    d_x = config.d_x
    d_p = config.d_p

    opt_flow = model.get_optimal_model()
    n_mc = 100

    plt.figure(figsize=(8, 6), dpi=80)
    for run in ["fragrant-sun-127"]:
        model, config, cnf = load_run_data(run)
        m_true = int(config.dataset_size / n_test)
        res_mc = []
        res_mc_gmlb = []
        for i in range(n_mc):
            _, mcrb_est_array, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m_true,
                                                      model,
                                                      norm_min=norm_min,
                                                      norm_max=norm_max,
                                                      run_optimal=True, run_lb=False, run_model=False)
            res_mc.append(mcrb_est_array)

            gmcrb_est_array, _, lb_array_z, norm_array = parameter_sweep(cnf, p_true, n_test, linear_ms, m,
                                                                         model,
                                                                         norm_max=norm_max,
                                                                         run_optimal=False)
            res_mc_gmlb.append(gmcrb_est_array)
        diag_array = torch.diagonal(torch.stack(res_mc), dim1=2, dim2=3).sum(dim=-1) / d_p
        diag_gmlb_array = torch.diagonal(torch.stack(res_mc_gmlb), dim1=2, dim2=3).sum(dim=-1) / d_p

        min_limit = config.min_limit
        # model_name = f"LTG-" + r"$a=$" + f"{min_limit}"
        # if 'ModelName.LinearGaussian' == config.model_name:
        model_name = ""
        plt.semilogy(norm_array.reshape([1, -1]).repeat([n_mc, 1]).numpy().flatten(), diag_array.flatten(), "o",
                     label=r"$\overline{LB}$", color="red")

        plt.semilogy(norm_array.reshape([1, -1]).repeat([n_mc, 1]).numpy().flatten(), diag_gmlb_array.flatten(), "x",
                     label=r"$GMLB$", color="green")
        plt.semilogy(norm_array.detach().numpy(),
                     (torch.diagonal(lb_array_z, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(),
                     label=f"LB", color="blue")

    plt.legend()
    plt.grid()
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\frac{\mathrm{Tr}(LB)}{d_p}$")
    plt.tight_layout()
    plt.savefig("compare.svg")
    plt.show()
    results_p_size = {}
    n_test = 100
    for p in DATASET_SIZE2RUNNAME.keys():
        # if p == 4:
        #     continue
        mean_re_list = []
        gmean_re_list = []
        std_re_list = []
        gstd_re_list = []

        run_name = DATASET_SIZE2RUNNAME[p][MAX_DATASET_SIZE]
        model, config, _ = load_run_data(run_name)
        p_true = model.parameters.random_sample_parameters()
        opt_flow = model.get_optimal_model()

        h_delta, l_delta = create_model_delta(config.d_x, config.d_p)
        h, c_xx = get_h_and_c_xx(model)
        l_x = torch.linalg.cholesky(c_xx)
        linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)

        # mcrb_est_array, lb_array_z = parameter_sweep(opt_flow, p_true, n_test_points, linear_ms, 128000, mcrb,
        #                                              h)

        for dataset_size in DATASET_SIZE2RUNNAME[p].keys():
            print(dataset_size)
            samples_per_point = int(dataset_size / n_test)

            run_name = DATASET_SIZE2RUNNAME[p][dataset_size]
            _, _, cnf = load_run_data(run_name)
            mcrb_mc_list = []
            gmcrb_mc_list = []
            for _ in tqdm(range(20)):
                mcrb_est_array, _, lb_array_z, _ = parameter_sweep(opt_flow, p_true, n_test, linear_ms,
                                                                   samples_per_point,
                                                                   model, norm_max=norm_max, norm_min=norm_min)
                gmcrb_est_array, _, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m, model,
                                                           norm_max=norm_max, norm_min=norm_min)
                mcrb_mc_list.append(mcrb_est_array)
                gmcrb_mc_list.append(gmcrb_est_array)
            # re = torch.norm(mcrb_est_array - lb_array_z, dim=(1, 2)) / torch.norm(lb_array_z, dim=(1, 2))
            re = torch.mean(torch.norm(torch.stack(mcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
                                       dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
            gre = torch.mean(torch.norm(torch.stack(gmcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
                                        dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
            # gre = torch.norm(gmcrb_est_array - lb_array_z, dim=(1, 2)) / torch.norm(lb_array_z, dim=(1, 2))
            mean_re_list.append(torch.mean(re).item())
            gmean_re_list.append(torch.mean(gre).item())
            std_re_list.append(torch.std(re).item())
            gstd_re_list.append(torch.std(gre).item())
        results_p_size.update({p: [mean_re_list, gmean_re_list, std_re_list, gstd_re_list]})
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    for p, r in results_p_size.items():
        if p != 4:
            ax1.errorbar(list(DATASET_SIZE2RUNNAME[p].keys()), r[0], fmt="--",
                         label=r"$\overline{\mathrm{LB}}$ $d_p$=" + f"{p}", yerr=r[2])
            ax1.errorbar(list(DATASET_SIZE2RUNNAME[p].keys()), r[1], label=r"$\mathrm{GMLB}$ $d_p$=" + f"{p}",
                         yerr=r[3])

    plt.legend()
    plt.xlabel("Dataset-size")
    plt.ylabel("xRE")
    plt.grid()
    plt.tight_layout()
    plt.savefig("dataset-size-effect.svg")
    plt.show()
