import torch
import numpy as np
import pyresearchutils as pru
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one, parameter_sweep
from experiments.analysis.helpers import load_run_data, create_model_delta, get_h_and_c_xx

from tqdm import tqdm

SEED = 2
RUNS_DICT = {3: {200: "jumping-music-248",
                 2000: "comic-firefly-242",
                 20000: "jolly-serenity-241",
                 200000: "effortless-glade-240"},
             4: {200: "polished-aardvark-238",
                 2000: "gentle-pine-236",
                 20000: "gallant-sky-235",
                 200000: "fluent-silence-234"},
             5: {200: "treasured-sun-230",
                 2000: "fresh-lake-229",
                 20000: "jolly-field-228",
                 200000: "dazzling-energy-220"}}

if __name__ == '__main__':
    pru.set_seed(SEED)
    # run_name = "dazzling-energy-220"
    alpha = 0.1
    beta = 0.1
    n_test = 20
    trimming_enable = False
    generate_delta = True
    run_interpolation_plot = True
    norm_min = 0.1
    norm_max = 9
    m = 640000
    if run_interpolation_plot:
        mc_n = 100
        results_dict = {}
        for run_name in ["visionary-cosmos-286", "proud-dragon-287"]:  # visionary-cosmos-286
            model, run_parameters, cnf = load_run_data(run_name)
            m_true = int(run_parameters.dataset_size / n_test)
            if generate_delta:
                generate_delta = False
                h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
                h, c_xx_overline = get_h_and_c_xx(model)
                l_x = torch.linalg.cholesky(c_xx_overline)
                linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
                p_true = model.parameters.random_sample_parameters()
            min_limit = None if not trimming_enable else run_parameters.min_limit * torch.ones(run_parameters.d_x,
                                                                                               device=pru.get_working_device())
            max_limit = None if not trimming_enable else run_parameters.max_limit * torch.ones(run_parameters.d_x,
                                                                                               device=pru.get_working_device())
            mc_mcrb = []
            mc_gmcrb = []
            for i in range(mc_n):
                run_lb = i == 0
                gmcrb_est_array, _, lb_array_z, norm_array = parameter_sweep(cnf, p_true, n_test,
                                                                             linear_ms, m,
                                                                             model,
                                                                             norm_min=norm_min,
                                                                             norm_max=norm_max,
                                                                             run_optimal=False,
                                                                             run_model=True,
                                                                             run_lb=run_lb,
                                                                             non_linear=run_parameters.non_linear_function,
                                                                             min_limit=min_limit,
                                                                             max_limit=max_limit)
                _, mcrb, _, _ = parameter_sweep(cnf, p_true, n_test,
                                                linear_ms, m_true,
                                                model,
                                                norm_min=norm_min,
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
            results_dict.update({run_parameters.max_limit: (pru.torch2numpy(diag_array).flatten(),
                                                            pru.torch2numpy(diag_gmlb_array).flatten(),
                                                            pru.torch2numpy(
                                                                torch.diagonal(lb_final, dim1=1, dim2=2).sum(
                                                                    dim=-1) / run_parameters.d_p))})
        import pickle

        #
        file_name = f"data_interpolation_reg_update_seed_{SEED}"
        file_name = f"{file_name}_trimming" if trimming_enable else file_name
        with open(f'../data/{file_name}.pickle', 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for max_limit, r in results_dict.items():
            plt.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                         r[0],
                         "o",
                         label=r"$\overline{LB}$ " f"c={max_limit}")
            plt.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                         r[1],
                         "x",
                         label=r"GMLB " + f"c={max_limit}")
            plt.semilogy(pru.torch2numpy(norm_array.detach()),
                         r[2],
                         label=f"LB c={max_limit}")

        ax = plt.gca()
        axins = ax.inset_axes([0.58, 0.05, 0.4, 0.4])
        for max_limit, r in results_dict.items():
            axins.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                           r[0],
                           "o",
                           label=r"$\overline{LB}$ " f"b=-a={max_limit}")
            axins.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                           r[1],
                           "x",
                           label=r"GMLB " + f"b=-a={max_limit}")
            axins.semilogy(pru.torch2numpy(norm_array.detach()),
                           r[2],
                           label=f"LB b=-a={max_limit}")
        axins.set_xlim(1.8, 2.6)
        axins.set_ylim(6, 10)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        ax.indicate_inset_zoom(axins, edgecolor="black")
        plt.grid()
        plt.legend()
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\frac{\mathrm{Tr}(LB)}{d_p}$")
        plt.tight_layout()
        plt.savefig("compare_nltn.svg")
        plt.show()
