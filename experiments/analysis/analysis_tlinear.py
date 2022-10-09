import torch
import numpy as np
import pyresearchutils as pru
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one, parameter_sweep
from experiments.analysis.helpers import load_run_data, create_model_delta, get_h_and_c_xx

from tqdm import tqdm

if __name__ == '__main__':
    pru.set_seed(0)
    run_name = "earnest-disco-159"
    alpha = 0.1
    beta = 0.1
    n_test = 20
    d_p = 8
    generate_delta = True
    run_interpolation_plot = False
    norm_max = 9
    m = 1280000
    if run_interpolation_plot:
        for run_name in ["treasured-surf-177", "bright-aardvark-180", "dandy-dream-181"]:
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
                                                                                      norm_max=norm_max,
                                                                                      run_optimal=True,
                                                                                      non_linear=run_parameters.non_linear_function)

            plt.semilogy(norm_array.detach().numpy(),
                         (torch.diagonal(lb_array_z, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(),
                         label=f"LB-a={run_parameters.min_limit}")
            plt.semilogy(norm_array.detach().numpy(),
                         (torch.diagonal(mcrb_est_array, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(), "o",
                         label=f"GMLB (Optimal)-a={run_parameters.min_limit}")
            plt.semilogy(norm_array.detach().numpy(),
                         (torch.diagonal(gmcrb_est_array, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(), "x",
                         label=f"GMLB (Trained)-a={run_parameters.min_limit}")
        plt.grid()
        plt.legend()
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\frac{\mathrm{Tr}(LB)}{d_p}$")
        plt.tight_layout()
        plt.savefig("compare_nltn.svg")
        plt.show()

    results = []
    dataset_size = []
    for run_name in ["treasured-surf-177", "blooming-dawn-187", "smooth-fire-188"]:
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
        for _ in tqdm(range(3)):
            _, mcrb_est_array, lb_array_z, _ = parameter_sweep(None, p_true, n_test, linear_ms,
                                                               samples_per_point,
                                                               model, norm_max=norm_max, run_optimal=True,
                                                               run_model=False,
                                                               non_linear=run_parameters.non_linear_function)
            gmcrb_est_array, _, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m, model,
                                                       norm_max=norm_max, run_optimal=False, run_model=True,
                                                       non_linear=run_parameters.non_linear_function)
            mcrb_mc_list.append(mcrb_est_array)
            gmcrb_mc_list.append(gmcrb_est_array)
        re = torch.mean(torch.norm(torch.stack(mcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
                                   dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
        gre = torch.mean(torch.norm(torch.stack(gmcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
                                    dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
        del mcrb_mc_list, mcrb_est_array, gmcrb_mc_list, gmcrb_est_array

        results.append([torch.mean(re).item(), torch.mean(gre).item()])
    # print("a")
    plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 0], label=r"$\overline{\mathrm{LB}}$")
    plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 1], label=r"$\mathrm{GMLB}$")
    plt.legend()
    plt.xlabel("Dataset-size")
    plt.ylabel("xRE")
    plt.grid()
    plt.tight_layout()
    plt.savefig("dataset-size-effect-nl.svg")
    plt.show()
