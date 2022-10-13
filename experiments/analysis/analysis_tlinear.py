import torch
import numpy as np
import pyresearchutils as pru
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one, parameter_sweep
from experiments.analysis.helpers import load_run_data, create_model_delta, get_h_and_c_xx

from tqdm import tqdm

if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "dazzling-energy-220"
    alpha = 0.1
    beta = 0.1
    n_test = 20

    generate_delta = True
    run_interpolation_plot = True
    norm_max = 9
    mc_n = 100
    m = 640000
    # "dazzling-energy-220", "fluent-silence-234","effortless-glade-240"
    # "warm-glade-253", "effortless-glade-240"
    if run_interpolation_plot:
        for run_name in ["comic-morning-261"]:
            model, run_parameters, cnf = load_run_data(run_name)
            m_true = int(run_parameters.dataset_size / 20)
            if generate_delta:
                generate_delta = False
                h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
                h, c_xx_overline = get_h_and_c_xx(model)
                l_x = torch.linalg.cholesky(c_xx_overline)
                linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
                p_true = model.parameters.random_sample_parameters()
            mc_mcrb = []
            mc_gmcrb = []
            for i in range(mc_n):
                run_lb = i == 0
                gmcrb_est_array, _, lb_array_z, norm_array = parameter_sweep(cnf, p_true, n_test,
                                                                             linear_ms, m,
                                                                             model,
                                                                             norm_max=norm_max,
                                                                             run_optimal=False,
                                                                             run_model=True,
                                                                             run_lb=run_lb,
                                                                             non_linear=run_parameters.non_linear_function)
                _, mcrb, _, _ = parameter_sweep(cnf, p_true, n_test,
                                                linear_ms, m_true,
                                                model,
                                                norm_max=norm_max,
                                                run_optimal=True,
                                                run_model=False,
                                                run_lb=False,
                                                non_linear=run_parameters.non_linear_function)
                mc_gmcrb.append(gmcrb_est_array)
                mc_mcrb.append(mcrb)
                if run_lb:
                    lb_final = lb_array_z
            diag_array = torch.diagonal(torch.stack(mc_mcrb), dim1=2, dim2=3).sum(dim=-1) / run_parameters.d_p
            diag_gmlb_array = torch.diagonal(torch.stack(mc_gmcrb), dim1=2, dim2=3).sum(dim=-1) / run_parameters.d_p

            plt.semilogy(norm_array.reshape([1, -1]).repeat([mc_n, 1]).numpy().flatten(), diag_array.flatten(), "o",
                         label=r"$\overline{LB}$")
            plt.semilogy(norm_array.reshape([1, -1]).repeat([mc_n, 1]).numpy().flatten(), diag_gmlb_array.flatten(),
                         "x",
                         label=r"$GMLB$")
            plt.semilogy(norm_array.detach().numpy(),
                         (torch.diagonal(lb_final, dim1=1, dim2=2).sum(dim=-1) / run_parameters.d_p).detach().numpy(),
                         label=f"LB-a={run_parameters.min_limit}")
        plt.grid()
        plt.legend()
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\frac{\mathrm{Tr}(LB)}{d_p}$")
        plt.tight_layout()
        plt.savefig("compare_nltn.svg")
        plt.show()
    n_test = 100
    results = []
    dataset_size = []
    for run_name in ["treasured-sun-230", "fresh-lake-229", "jolly-field-228",
                     "dazzling-energy-220"]:  # ["treasured-surf-177", "blooming-dawn-187", "smooth-fire-188"]:
        model, run_parameters, cnf = load_run_data(run_name)
        samples_per_point = int(run_parameters.dataset_size / n_test)
        dataset_size.append(run_parameters.dataset_size)
        if generate_delta:
            generate_delta = False
            h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
            h, c_xx_overline = get_h_and_c_xx(model)
            l_x = torch.linalg.cholesky(c_xx_overline)
            linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
            p_true = model.parameters.random_sample_parameters()

        mcrb_mc_list = []
        gmcrb_mc_list = []
        for _ in tqdm(range(1)):
            _, mcrb_est_array, lb_array_z, _ = parameter_sweep(None, p_true, n_test, linear_ms,
                                                               samples_per_point,
                                                               model, norm_max=norm_max, run_optimal=True,
                                                               run_model=False,
                                                               run_lb=True,
                                                               non_linear=run_parameters.non_linear_function)
            gmcrb_est_array, _, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m, model,
                                                       norm_max=norm_max, run_optimal=False, run_model=True,
                                                       run_lb=False,
                                                       non_linear=run_parameters.non_linear_function)
            mcrb_mc_list.append(mcrb_est_array)
            gmcrb_mc_list.append(gmcrb_est_array)
        re = torch.mean(torch.norm(torch.stack(mcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
                                   dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
        gre = torch.mean(torch.norm(torch.stack(gmcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
                                    dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)

        results.append([torch.mean(re).item(), torch.mean(gre).item()])
    # print("a")
    # print(np.asarray(results)[:, 0])
    # print(np.asarray(results)[:, 1])
    plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 0], label=r"$\overline{\mathrm{LB}}$")
    plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 1], label=r"$\mathrm{GMLB}$")
    plt.legend()
    plt.xlabel("Dataset-size")
    plt.ylabel("xRE")
    plt.grid()
    plt.tight_layout()
    plt.savefig("dataset-size-effect-nl.svg")
    plt.show()
