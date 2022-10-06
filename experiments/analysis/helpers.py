import torch
import wandb
import os
from experiments import constants
from experiments.mcrb.linear_mcrb import LinearMCRB
from experiments import measurements_distributions
from experiments import flow_models
from argparse import Namespace
import gmlb
import copy

FLOW_BEST = "flow_best.pt"


def parameter_sweep(in_flow, in_p_true, in_n_test_points, in_linear_ms, in_samples_per_point, in_mcrb, in_h):
    norm_array = torch.linspace(0.1, 3, in_n_test_points)
    res_list = []
    lb_list = []
    _p_true = copy.copy(in_p_true)
    for norm in norm_array:
        _p_true[constants.THETA] = in_p_true[constants.THETA] / torch.norm(in_p_true[constants.THETA])
        _p_true[constants.THETA] = _p_true[constants.THETA] * norm

        _p_zero = in_linear_ms.calculate_pseudo_true_parameter(torch.matmul(in_h, _p_true[constants.THETA].flatten()))
        _lb = gmlb.compute_lower_bound(in_mcrb, _p_true[constants.THETA].flatten(), _p_zero)

        _, _gmlb_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(in_flow,
                                                                           in_samples_per_point,
                                                                           in_linear_ms, **_p_true)
        res_list.append(_gmlb_v)
        lb_list.append(_lb)
    return torch.stack(res_list), torch.stack(lb_list)


def create_model_delta(in_d_x, in_d_p, scale=0.1):
    _h_delta = torch.randn(in_d_x, in_d_p) * scale
    c_vv_delta = torch.randn(in_d_x, in_d_x) * scale
    c_vv_delta = torch.matmul(c_vv_delta, c_vv_delta.T)
    _l_delta = torch.linalg.cholesky(c_vv_delta)
    return _h_delta, _l_delta


def download_file(in_run, in_file):
    if os.path.isfile(in_file):
        os.remove(in_file)
    in_run.file(in_file).download()


def build_misspecifietion_type_one(in_h, in_l_xx, in_h_delta, in_l_vv_delta, in_alpha, in_beta):
    h_matrix = in_h + in_alpha * in_h_delta
    l_c = in_l_xx + in_beta * in_l_vv_delta
    c_ww_matrix = torch.matmul(l_c, l_c.T)
    return LinearMCRB(h_matrix, c_ww_matrix)


def load_run_data(in_run_name):
    api = wandb.Api()
    runs = api.runs(f"HVH/{constants.PROJECT}")
    for run in runs:
        print(run.name, run.state)
        if run.name == in_run_name:
            download_file(run, FLOW_BEST)
            run_parameters = Namespace(**run.config)

            model_name = measurements_distributions.ModelName[run.config['model_name'].split(".")[-1]]
            _model = measurements_distributions.get_measurement_distribution(
                model_name,
                d_x=run_parameters.d_x,
                d_p=run_parameters.d_p,
                norm_min=run_parameters.norm_min,
                norm_max=run_parameters.norm_max,
                a_limit=float(run_parameters.min_limit),
                b_limit=float(run_parameters.max_limit))
            _cnf = flow_models.generate_cnf_model(run_parameters.d_x,
                                                  run_parameters.d_p,
                                                  [constants.THETA],
                                                  n_blocks=run_parameters.n_blocks,
                                                  n_layer_inject=run_parameters.n_layer_inject,
                                                  n_hidden_inject=run_parameters.n_hidden_inject,
                                                  inject_scale=run_parameters.inject_scale,
                                                  inject_bias=run_parameters.inject_bias
                                                  )
            data = torch.load(FLOW_BEST, map_location="cpu")
            _cnf.load_state_dict(data)
            download_file(run, _model.file_name)
            _model.load_data_model("./")
            return _model, run_parameters, _cnf
