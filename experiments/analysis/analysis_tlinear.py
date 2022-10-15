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
    # "dazzling-energy-220", "fluent-silence-234","effortless-glade-240"
    # "warm-glade-253", "effortless-glade-240" breezy-snowflake-266
    if run_interpolation_plot:
        mc_n = 100
        results_dict = {}
        for run_name in ["effortless-glade-240", "dazzling-energy-220"]:
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
            results_dict.update({run_parameters.max_limit: (pru.torch2numpy(diag_array).flatten(),
                                                            pru.torch2numpy(diag_gmlb_array).flatten(),
                                                            pru.torch2numpy(
                                                                torch.diagonal(lb_final, dim1=1, dim2=2).sum(
                                                                    dim=-1) / run_parameters.d_p))})
        for max_limit, r in results_dict.items():
            plt.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                         r[0],
                         "o",
                         label=r"$\overline{LB}$ " f"b=-a={max_limit}")
            plt.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                         r[1],
                         "x",
                         label=r"GMLB " + f"b=-a={max_limit}")
            plt.semilogy(pru.torch2numpy(norm_array.detach()),
                         r[2],
                         label=f"LB b=-a={max_limit}")

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
#     n_test = 100
#     results = []
#     dataset_size = []
#     for run_name in RUNS_DICT[5].values():  # ["treasured-surf-177", "blooming-dawn-187", "smooth-fire-188"]:
#         model, run_parameters, cnf = load_run_data(run_name)
#         samples_per_point = int(run_parameters.dataset_size / n_test)
#         dataset_size.append(run_parameters.dataset_size)
#         if generate_delta:
#             generate_delta = False
#             h_delta, l_delta = create_model_delta(run_parameters.d_x, run_parameters.d_p)
#             h, c_xx_overline = get_h_and_c_xx(model)
#             l_x = torch.linalg.cholesky(c_xx_overline)
#             linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
#             p_true = model.parameters.random_sample_parameters()
#
#         mcrb_mc_list = []
#         gmcrb_mc_list = []
#         for _ in tqdm(range(1)):
#             _, mcrb_est_array, lb_array_z, _ = parameter_sweep(None, p_true, n_test, linear_ms,
#                                                                samples_per_point,
#                                                                model, norm_max=norm_max, run_optimal=True,
#                                                                run_model=False,
#                                                                run_lb=True,
#                                                                non_linear=run_parameters.non_linear_function)
#             gmcrb_est_array, _, _, _ = parameter_sweep(cnf, p_true, n_test, linear_ms, m, model,
#                                                        norm_max=norm_max, run_optimal=False, run_model=True,
#                                                        run_lb=False,
#                                                        non_linear=run_parameters.non_linear_function)
#             mcrb_mc_list.append(mcrb_est_array)
#             gmcrb_mc_list.append(gmcrb_est_array)
#         re = torch.mean(torch.norm(torch.stack(mcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
#                                    dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
#         gre = torch.mean(torch.norm(torch.stack(gmcrb_mc_list) - torch.unsqueeze(lb_array_z, dim=0),
#                                     dim=(2, 3)) / torch.unsqueeze(torch.norm(lb_array_z, dim=(1, 2)), dim=0), dim=1)
#
#         results.append([torch.mean(re).item(), torch.mean(gre).item()])
#     # print("a")
#     # print(np.asarray(results)[:, 0])
#     # print(np.asarray(results)[:, 1])
#     plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 0], label=r"$\overline{\mathrm{LB}}$")
#     plt.semilogx(np.asarray(dataset_size), np.asarray(results)[:, 1], label=r"$\mathrm{GMLB}$")
#     plt.legend()
#     plt.xlabel("Dataset-size")
#     plt.ylabel("xRE")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig("dataset-size-effect-nl.svg")
#     plt.show()
#
#     fig, ax1 = plt.subplots(1, 1)
#     ax1.set_xscale("log")
#     ax1.set_yscale("log")
#     for p, r in results_p_size.items():
#         if p != 4:
#             ax1.errorbar(list(DATASET_SIZE2RUNNAME[p].keys()), r[0], fmt="--",
#                          label=r"$\overline{\mathrm{LB}}$ $d_p$=" + f"{p}", yerr=r[2])
#             ax1.errorbar(list(DATASET_SIZE2RUNNAME[p].keys()), r[1], label=r"$\mathrm{GMLB}$ $d_p$=" + f"{p}",
#                          yerr=r[3])
#
#     plt.legend()
#     plt.xlabel("Dataset-size")
#     plt.ylabel("xRE")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig("dataset-size-effect.svg")
#     plt.show()
#     # print("---")
#     # print(h_delta)
#
#     # mc = pru.MetricCollector()
#     # p_true_iter = copy.copy(p_true)
#     # for scale in alpha_array:
#     #     p_true_iter[constants.THETA] = p_true[constants.THETA] * scale / torch.norm(p_true[constants.THETA])
#     #
#     #     # mu_overline = soft_clip(torch.matmul(p_true_iter[constants.THETA], h.T), torch.min(model.a),
#     #     #                         torch.max(model.b))
#     #     mu_overline = torch.matmul(p_true_iter[constants.THETA], h.T)
#     #
#     #     mu, c_xx = compute_mean_covarinace(model, mu_overline)
#     #
#     #     mcrb = linear_ms.calculate_mcrb(0, c_xx)
#     #     p_zero = linear_ms.calculate_pseudo_true_parameter(mu.flatten())
#     #     lb = gmlb.compute_lower_bound(mcrb, p_true_iter[constants.THETA].flatten(), p_zero)
#     #
#     #     gmcrb, gmlb_v, p_zero_est = gmlb.generative_misspecified_cramer_rao_bound(model.generate_data, 256000,
#     #                                                                               linear_ms,
#     #                                                                               **p_true_iter)
#     #     gmcrb_cnf, gmlb_cnf_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 256000,
#     #                                                                                   linear_ms,
#     #                                                                                   min_limit=model.a,
#     #                                                                                   max_limit=model.b,
#     #                                                                                   **p_true_iter)
#     #
#     #     mc.insert(
#     #         lb=torch.trace(lb).item() / run_parameters.d_p,
#     #         crb=torch.trace(linear_ms.crb()) / run_parameters.d_p,
#     #         mcrb=torch.trace(mcrb).item() / run_parameters.d_p,
#     #         gmcrb=torch.trace(gmcrb).item() / run_parameters.d_p,
#     #         gmcrb_cnf=torch.trace(gmcrb_cnf).item() / run_parameters.d_p,
#     #         gmlb_cnf=torch.trace(gmlb_cnf_v).item() / run_parameters.d_p,
#     #         gmlb=torch.trace(gmlb_v).item() / run_parameters.d_p)
# #
# #     plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label=f"GMLB (Optimal) $a=$" + f"{run_parameters.min_limit}")
# #     plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x",
# #              label=f"GMLB (Trained) $a=$" + f"{run_parameters.min_limit}")
# #     plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label=f"LB $a=$" + f"{run_parameters.min_limit}")
# # plt.grid()
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("trunced_res.svg")
# # plt.show()
