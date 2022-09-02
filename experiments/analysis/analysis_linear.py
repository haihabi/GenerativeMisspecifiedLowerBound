import numpy as np
import torch

from experiments import constants
from experiments import measurements_distributions
import gmlb
from matplotlib import pyplot as plt
import pyresearchutils as pru
import wandb
import os
from experiments import flow_models
from experiments.analysis.helpers import build_misspecifietion_type_one

FLOW_BEST = "flow_best.pt"


def get_h_and_c_xx(in_opt_flow):
    _h = in_opt_flow.flow.flows[0].h
    _c_xx = in_opt_flow.flow.flows[0].c_xx
    return _h, _c_xx


def download_file(in_run, in_file):
    if os.path.isfile(in_file):
        os.remove(in_file)
    in_run.file(in_file).download()


def load_run_data(in_run_name):
    api = wandb.Api()
    runs = api.runs(f"HVH/{constants.PROJECT}")
    for run in runs:
        print(run.name, run.state)
        if run.name == in_run_name:
            download_file(run, FLOW_BEST)
            _model = measurements_distributions.LinearModel(run.config["d_x"], run.config["d_p"],
                                                            run.config["norm_min"],
                                                            run.config["norm_max"])
            _cnf = flow_models.generate_cnf_model(run.config["d_x"], run.config["d_p"], [constants.THETA])
            data = torch.load(FLOW_BEST, map_location="cpu")
            _cnf.load_state_dict(data)
            download_file(run, _model.file_name)
            _model.load_data_model("./")
            return _model, run.config, _cnf


if __name__ == '__main__':
    pru.set_seed(0)
    # run_name = "woven-snow-50"
    run_name = "ethereal-meadow-51"
    model, config, cnf = load_run_data(run_name)
    dataset_size = config["dataset_size"]
    d_x = config["d_x"]
    d_p = config["d_p"]

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
    alpha_array = np.linspace(-0.5, 0.5, n_test)

    mc = pru.MetricCollector()

    mc.clear()
    for alpha in alpha_array:
        linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.3)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)
        mu = torch.matmul(h, p_true[constants.THETA].flatten())
        p_zero = linear_ms.calculate_pseudo_true_parameter(mu)

        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound_flow(opt_flow, 50000, linear_ms, **p_true)
        gmcrb_cnf, gmlb_cnf = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 50000, linear_ms, **p_true)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  crb=torch.trace(linear_ms.crb()) / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmcrb_cnf=torch.trace(gmcrb_cnf).item() / d_p,
                  gmlb_cnf=torch.trace(gmlb_cnf).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 2, 1)
    plt.plot(alpha_array, np.asarray(mc["mcrb"]), label="MCRB")
    plt.plot(alpha_array, np.asarray(mc["gmcrb"]), "o", label="GMCRB (Optimal)")
    plt.plot(alpha_array, np.asarray(mc["gmcrb_cnf"]), "x", label="GMCRB (CNF)")
    plt.xlabel(r"$\alpha$")
    # plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")
    # plt.plot(alpha_array, np.asarray(mc["crb"]), label="CRB (Assume)")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(alpha_array, np.asarray(mc["lb"]), label="LB")
    plt.plot(alpha_array, np.asarray(mc["gmlb"]), "o", label="GMLB (Optimal)")
    plt.plot(alpha_array, np.asarray(mc["gmlb_cnf"]), "x", label="GMLB (CNF)")
    # plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")
    # plt.plot(alpha_array, np.asarray(mc["crb"]), label="CRB (Assume)")

    plt.legend()
    plt.grid()
    plt.xlabel(r"$\alpha$")
    plt.savefig("compare.svg")
    plt.show()

    p_true = model.parameters.random_sample_parameters()

    n_test_points = 200
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, 0.1, 0.0)
    mcrb = linear_ms.calculate_mcrb(h, c_xx)
    k = int(dataset_size / n_test_points)
    mc.clear()
    norm_array = torch.linspace(0.1, 3, n_test_points)
    for norm in norm_array:
        p_true[constants.THETA] = p_true[constants.THETA] / torch.norm(p_true[constants.THETA])
        p_true[constants.THETA] = p_true[constants.THETA] * norm
        mu = torch.matmul(h, p_true[constants.THETA].flatten())
        p_zero = linear_ms.calculate_pseudo_true_parameter(mu)
        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound_flow(opt_flow, int(dataset_size / n_test_points),
                                                                           linear_ms, **p_true)
        gmcrb_cnf, gmlb_cnf = gmlb.generative_misspecified_cramer_rao_bound_flow(cnf, 64000,
                                                                                 linear_ms, **p_true)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  crb=torch.trace(linear_ms.crb()) / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmcrb_cnf=torch.trace(gmcrb_cnf).item() / d_p,
                  gmlb_cnf=torch.trace(gmlb_cnf).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)
    plt.subplot(1, 2, 1)
    plt.plot(norm_array.numpy(), mc["lb"], label="LB")
    plt.plot(norm_array.numpy(), mc["gmlb_cnf"], "o", label="GMLB (CNF)")
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(norm_array.numpy(), mc["lb"], label="LB")
    plt.plot(norm_array.numpy(), mc["gmlb"], "o", label=f"ELB ({k} Sample Per Point)")
    plt.grid()
    plt.legend()
    plt.savefig("dataset_size_ill.svg")
    plt.show()
    print("a")
    # plt.plot(alpha_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    # plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label="LB")
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # mc.clear()
    # beta_array = np.linspace(-1.5, 1.5, 100)
    # for beta in beta_array:
    #     linear_ms = build_misspecifietion_type_one(h, c_xx, h_delta, c_vv_delta, 0.0, beta)
    #     mcrb = linear_ms.calculate_mcrb(h, c_xx)
    #     p_zero = linear_ms.calculate_pseudo_true_parameter(p_true[constants.THETA].flatten(), h)
    #     lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
    #     gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(opt_flow, 50000, linear_ms, **p_true)
    #     mc.insert(lb=torch.trace(lb).item() / d_p,
    #               mcrb=torch.trace(mcrb).item() / d_p,
    #               gmcrb=torch.trace(gmcrb).item() / d_p,
    #               gmlb=torch.trace(gmlb_v).item() / d_p)
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(beta_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    # plt.plot(beta_array, np.asarray(mc["gmcrb"]), "-o", label="GMCRB (Optimal)")
    # plt.plot(beta_array, np.ones(len(beta_array)) * true_crb, label="CRB (True)")
    # plt.legend()
    # plt.grid()
    # plt.subplot(1, 2, 2)
    # plt.plot(beta_array, np.asarray(mc["lb"]), "--", label="LB")
    # plt.plot(beta_array, np.asarray(mc["gmlb"]), "-o", label="GMLB (Optimal)")
    # plt.legend()
    #
    # plt.grid()
    # plt.show()
