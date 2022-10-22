import torch
import numpy as np
import pyresearchutils as pru
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one, parameter_sweep
from experiments.analysis.helpers import load_run_data, create_model_delta, get_h_and_c_xx

from tqdm import tqdm

# RUNS_DICT = {3: {200: "jumping-music-248",
#                  2000: "comic-firefly-242",
#                  20000: "jolly-serenity-241",
#                  200000: "effortless-glade-240"},
#              4: {200: "polished-aardvark-238",
#                  2000: "gentle-pine-236",
#                  20000: "gallant-sky-235",
#                  200000: "fluent-silence-234"},
#              5: {200: "treasured-sun-230",
#                  2000: "fresh-lake-229",
#                  20000: "jolly-field-228",
#                  200000: "dazzling-energy-220"}}

# With Regularization
RUNS_DICT = {
    # 3: {200: "stellar-cosmos-290",
    #     2000: "likely-firebrand-289",
    #     20000: "visionary-cosmos-286",
    #     200000: "exalted-yogurt-288"},
    5: {200: "divine-salad-291",
        2000: "wandering-darkness-292",
        20000: "proud-dragon-287",
        200000: "logical-wave-293"}}

if __name__ == '__main__':
    pru.set_seed(2)
    alpha = 0.1
    beta = 0.1
    generate_delta = True
    trimming_enable = True
    norm_max = 9
    m = 64000
    n_test = 100
    mc_n = 20
    results_dict = {}
    for a in RUNS_DICT.keys():
        results_size_dict = {}
        for j, run_name in enumerate(RUNS_DICT[a].values()):
            model, run_parameters, cnf = load_run_data(run_name, affine_inject_base=False)
            samples_per_point = int(run_parameters.dataset_size / n_test)
            if generate_delta:
                generate_delta = False
                h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
                h, c_xx_overline = get_h_and_c_xx(model)
                l_x = torch.linalg.cholesky(c_xx_overline)
                linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
                p_true = model.parameters.random_sample_parameters()
            # if trimming_enable:
            min_limit = None if not trimming_enable else run_parameters.min_limit * torch.ones(run_parameters.d_x,
                                                                                               device=pru.get_working_device())
            max_limit = None if not trimming_enable else run_parameters.max_limit * torch.ones(run_parameters.d_x,
                                                                                               device=pru.get_working_device())
            if j == 0:
                best_gmcrb_mc_list = []
                for i in tqdm(range(mc_n)):
                    _, mcrb_est_array, lb_array_z, _ = parameter_sweep(None, p_true, n_test, linear_ms,
                                                                       m,
                                                                       model, norm_max=norm_max, run_optimal=True,
                                                                       run_model=False,
                                                                       run_lb=i == 0,
                                                                       non_linear=run_parameters.non_linear_function)
                    if i == 0:
                        lb_array = lb_array_z
                    best_gmcrb_mc_list.append(mcrb_est_array)
                best_gre = torch.mean(torch.norm(torch.stack(best_gmcrb_mc_list) - torch.unsqueeze(lb_array, dim=0),
                                                 dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array, dim=(1, 2)), dim=0),
                                      dim=1)
                print("a")
            mcrb_mc_list = []
            gmcrb_mc_list = []
            for i in tqdm(range(mc_n)):
                _, mcrb_est_array, lb_array_z, _ = parameter_sweep(None, p_true, n_test, linear_ms,
                                                                   samples_per_point,
                                                                   model, norm_max=norm_max, run_optimal=True,
                                                                   run_model=False,
                                                                   run_lb=i == 0 and j == 0,
                                                                   non_linear=run_parameters.non_linear_function)
                gmcrb_est_array, _, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m, model,
                                                           norm_max=norm_max, run_optimal=False, run_model=True,
                                                           run_lb=False,
                                                           non_linear=run_parameters.non_linear_function,
                                                           min_limit=min_limit, max_limit=max_limit)
                if i == 0 and j == 0:
                    lb_array = lb_array_z
                mcrb_mc_list.append(mcrb_est_array)
                gmcrb_mc_list.append(gmcrb_est_array)
            re = torch.mean(torch.norm(torch.stack(mcrb_mc_list) - torch.unsqueeze(lb_array, dim=0),
                                       dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array, dim=(1, 2)), dim=0), dim=1)
            gre = torch.mean(torch.norm(torch.stack(gmcrb_mc_list) - torch.unsqueeze(lb_array, dim=0),
                                        dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array, dim=(1, 2)), dim=0), dim=1)

            results_size_dict.update({run_parameters.dataset_size: [torch.mean(re).item(), torch.mean(gre).item()]})
        results_dict.update({a: results_size_dict})

    import pickle

    with open(f'../data/data_reg_{n_test}_update_v2_seed_trimming.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
