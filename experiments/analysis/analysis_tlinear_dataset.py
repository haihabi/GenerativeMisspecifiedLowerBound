import torch
import numpy as np
import pyresearchutils as pru
from matplotlib import pyplot as plt
from experiments.analysis.helpers import build_misspecifietion_type_one, parameter_sweep
from experiments.analysis.helpers import load_run_data, create_model_delta, get_h_and_c_xx

from tqdm import tqdm

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
    pru.set_seed(0)
    # run_name = "dazzling-energy-220"
    alpha = 0.1
    beta = 0.1
    n_test = 20
    generate_delta = True
    run_interpolation_plot = True
    norm_max = 9
    m = 640000
    n_test = 100
    results_dict = {}
    for a in RUNS_DICT.keys():
        results = []
        dataset_size = []
        for run_name in RUNS_DICT[a].values():  # ["treasured-surf-177", "blooming-dawn-187", "smooth-fire-188"]:
            model, run_parameters, cnf = load_run_data(run_name, affine_inject_base=False)
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
            for i in tqdm(range(20)):
                _, mcrb_est_array, lb_array_z, _ = parameter_sweep(None, p_true, n_test, linear_ms,
                                                                   samples_per_point,
                                                                   model, norm_max=norm_max, run_optimal=True,
                                                                   run_model=False,
                                                                   run_lb=i == 0,
                                                                   non_linear=run_parameters.non_linear_function)
                gmcrb_est_array, _, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m, model,
                                                           norm_max=norm_max, run_optimal=False, run_model=True,
                                                           run_lb=False,
                                                           non_linear=run_parameters.non_linear_function)
                if i == 0:
                    lb_array = lb_array_z
                mcrb_mc_list.append(mcrb_est_array)
                gmcrb_mc_list.append(gmcrb_est_array)
            re = torch.mean(torch.norm(torch.stack(mcrb_mc_list) - torch.unsqueeze(lb_array, dim=0),
                                       dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array, dim=(1, 2)), dim=0), dim=1)
            gre = torch.mean(torch.norm(torch.stack(gmcrb_mc_list) - torch.unsqueeze(lb_array, dim=0),
                                        dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array, dim=(1, 2)), dim=0), dim=1)

            results.append([torch.mean(re).item(), torch.mean(gre).item()])
        results_dict.update({a: results})

    # print("a")
    # print(np.asarray(results)[:, 0])
    # print(np.asarray(results)[:, 1])
    # plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 0], label=r"$\overline{\mathrm{LB}}$")
    # plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 1], label=r"$\mathrm{GMLB}$")
    # plt.legend()
    # plt.xlabel("Dataset-size")
    # plt.ylabel("xRE")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("dataset-size-effect-nl.svg")
    # plt.show()
    import pickle

    # pickle.dumps(results_dict,"data.pickle")
    with open('../data/data.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # fig, ax1 = plt.subplots(1, 1)
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    # for a, r in results_dict.items():
    #     # if p != 4:
    #     r = np.asarray(r)
    #     ax1.errorbar(list(RUNS_DICT[a].keys()), r[:, 0], fmt="--",
    #                  label=r"$\overline{\mathrm{LB}}$ $a$=" + f"{a}")
    #     ax1.errorbar(list(RUNS_DICT[a].keys()), r[:, 1], label=r"$\mathrm{GMLB}$ $a$=" + f"{a}")
    #
    # plt.legend()
    # plt.xlabel("Dataset-size")
    # plt.ylabel("xRE")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("dataset-size-effect.svg")
    # plt.show()
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
#
#     plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label=f"GMLB (Optimal) $a=$" + f"{run_parameters.min_limit}")
#     plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x",
#              label=f"GMLB (Trained) $a=$" + f"{run_parameters.min_limit}")
#     plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label=f"LB $a=$" + f"{run_parameters.min_limit}")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("trunced_res.svg")
# plt.show()
