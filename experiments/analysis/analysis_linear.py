import numpy as np
import torch

from experiments import constants
from experiments.mcrb.linear_mcrb import LinearMCRB
from experiments import measurements_distributions
import gmlb
from matplotlib import pyplot as plt
import pyresearchutils as pru
import wandb
import os
from experiments import flow_models

FLOW_BEST = "flow_best.pt"


def get_h_and_c_xx(in_opt_flow):
    _h = in_opt_flow.flow.flows[0].h
    _c_xx = in_opt_flow.flow.flows[0].c_xx
    return _h, _c_xx


def build_misspecifietion_type_one(in_h, in_l_xx, in_h_delta, in_l_vv_delta, in_alpha, in_beta):
    h_matrix = in_h + in_alpha * in_h_delta
    l_c = in_l_xx + in_beta * in_l_vv_delta
    c_ww_matrix = torch.matmul(l_c, l_c.T)
    return LinearMCRB(h_matrix, c_ww_matrix)


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
    run_name = "resilient-sunset-36"
    model, config, cnf = load_run_data(run_name)
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
    alpha_array = np.linspace(-1.5, 1.5, 100)

    mc = pru.MetricCollector()

    mc.clear()
    for alpha in alpha_array:
        linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, 0.0)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)
        p_zero = linear_ms.calculate_pseudo_true_parameter(p_true[constants.THETA].flatten(), h)
        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(opt_flow, 50000, linear_ms, **p_true)
        gmcrb_cnf, gmlb_cnf = gmlb.generative_misspecified_cramer_rao_bound(cnf, 50000, linear_ms, **p_true)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmcrb_cnf=torch.trace(gmcrb_cnf).item() / d_p,
                  gmlb_cnf=torch.trace(gmlb_cnf).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)

    plt.subplot(1, 2, 1)
    plt.plot(alpha_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    plt.plot(alpha_array, np.asarray(mc["gmcrb"]), "-o", label="GMCRB (Optimal)")
    plt.plot(alpha_array, np.asarray(mc["gmcrb_cnf"]), "-x", label="GMCRB (CNF)")
    plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label="LB")
    plt.plot(alpha_array, np.asarray(mc["gmlb"]), "-o", label="GMLB (Optimal)")
    plt.plot(alpha_array, np.ones(len(alpha_array)) * true_crb, label="CRB (True)")

    plt.legend()

    plt.grid()
    plt.show()

    plt.plot(alpha_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    plt.plot(alpha_array, np.asarray(mc["lb"]), "--", label="LB")
    plt.grid()
    plt.legend()
    plt.show()

    mc.clear()
    beta_array = np.linspace(-1.5, 1.5, 100)
    for beta in beta_array:
        linear_ms = build_misspecifietion_type_one(h, c_xx, h_delta, c_vv_delta, 0.0, beta)
        mcrb = linear_ms.calculate_mcrb(h, c_xx)
        p_zero = linear_ms.calculate_pseudo_true_parameter(p_true[constants.THETA].flatten(), h)
        lb = gmlb.compute_lower_bound(mcrb, p_true[constants.THETA].flatten(), p_zero)
        gmcrb, gmlb_v = gmlb.generative_misspecified_cramer_rao_bound(opt_flow, 50000, linear_ms, **p_true)
        mc.insert(lb=torch.trace(lb).item() / d_p,
                  mcrb=torch.trace(mcrb).item() / d_p,
                  gmcrb=torch.trace(gmcrb).item() / d_p,
                  gmlb=torch.trace(gmlb_v).item() / d_p)

    plt.subplot(1, 2, 1)
    plt.plot(beta_array, np.asarray(mc["mcrb"]), "--", label="MCRB")
    plt.plot(beta_array, np.asarray(mc["gmcrb"]), "-o", label="GMCRB (Optimal)")
    plt.plot(beta_array, np.ones(len(beta_array)) * true_crb, label="CRB (True)")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(beta_array, np.asarray(mc["lb"]), "--", label="LB")
    plt.plot(beta_array, np.asarray(mc["gmlb"]), "-o", label="GMLB (Optimal)")
    plt.legend()

    plt.grid()
    plt.show()
