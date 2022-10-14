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
import pyresearchutils as pru
from experiments.measurements_distributions.linear_truncated_gaussian.computing_moments import \
    compute_second_order_state
from experiments.measurements_distributions.linear_truncated_gaussian.softclip import soft_clip
from tqdm import tqdm

FLOW_BEST = "flow_best.pt"


def get_h_and_c_xx(in_model):
    return in_model.h, in_model.c_xx_bar


def compute_mean_covarinace_truncated_norm(in_model, in_mu_overline):
    lb = in_model.a.detach().cpu().numpy()
    ub = in_model.b.detach().cpu().numpy()
    c_xx_bar = in_model.c_xx_bar.detach().cpu().numpy()
    _mu, _c_xx = compute_second_order_state(lb, ub, in_mu_overline.detach().cpu().numpy().flatten(), c_xx_bar)
    return torch.tensor(_mu).float(), torch.tensor(_c_xx).float()


def parameter_sweep(in_flow, in_p_true, in_n_test_points, in_linear_ms, in_samples_per_point, in_model,
                    norm_min=0.1, norm_max=10, non_linear=False, run_optimal=False, run_model=True, run_lb=True):
    norm_array = torch.linspace(norm_min, norm_max, in_n_test_points)
    res_list = []
    res_opt_list = []
    lb_list = []
    _p_true = copy.copy(in_p_true)
    h, c_xx = get_h_and_c_xx(in_model)
    for norm in tqdm(norm_array):
        _p_true[constants.THETA] = in_p_true[constants.THETA] / torch.norm(in_p_true[constants.THETA])
        _p_true[constants.THETA] = _p_true[constants.THETA] * norm
        if run_lb:
            mu = torch.matmul(h, _p_true[constants.THETA].flatten())
            if isinstance(in_model, measurements_distributions.TruncatedLinearModel):
                mu_overline = mu
                if non_linear:
                    mu_overline = soft_clip(mu_overline, torch.min(in_model.a), torch.min(in_model.b))
                mu_overline = torch.clip(mu_overline, min=torch.min(in_model.a), max=torch.min(in_model.b))
                mu, c_xx = compute_mean_covarinace_truncated_norm(in_model, mu_overline)
                mu = mu.to(pru.get_working_device())
                c_xx = c_xx.to(pru.get_working_device())

            if isinstance(in_model, measurements_distributions.NonLinearGaussian):
                mu = soft_clip(mu, torch.min(torch.ones(1) * in_model.a_limit),
                               torch.min(torch.ones(1) * in_model.b_limit))

            _mcrb = in_linear_ms.calculate_mcrb(0, c_xx)
            _p_zero = in_linear_ms.calculate_pseudo_true_parameter(mu)
            _lb = gmlb.compute_lower_bound(_mcrb, _p_true[constants.THETA].flatten(), _p_zero)
            lb_list.append(_lb)
        if run_model:
            _, _gmlb_v, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(in_flow,
                                                                               in_samples_per_point,
                                                                               in_linear_ms, **_p_true)
            res_list.append(_gmlb_v)
        if run_optimal:
            if isinstance(in_model, measurements_distributions.TruncatedLinearModel):
                _, _gmlb_v_optimal, _ = gmlb.generative_misspecified_cramer_rao_bound(in_model.generate_data,
                                                                                      in_samples_per_point,
                                                                                      in_linear_ms,
                                                                                      **_p_true)
            else:
                _, _gmlb_v_optimal, _ = gmlb.generative_misspecified_cramer_rao_bound_flow(in_model.get_optimal_model(),
                                                                                           in_samples_per_point,
                                                                                           in_linear_ms, **_p_true)
            res_opt_list.append(_gmlb_v_optimal)

    if len(res_list) > 0: res_list = torch.stack(res_list)
    if len(res_opt_list) > 0: res_opt_list = torch.stack(res_opt_list)
    if len(lb_list) > 0: lb_list = torch.stack(lb_list)

    return res_list, res_opt_list, lb_list, norm_array


def create_model_delta(in_d_x, in_d_p, scale=0.1):
    _h_delta = torch.randn(in_d_x, in_d_p, device=pru.get_working_device()) * scale
    c_vv_delta = torch.randn(in_d_x, in_d_x, device=pru.get_working_device()) * scale
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
            if run.config.get("non_linear_function") is None:
                run.config["non_linear_function"] = False
            run_parameters = Namespace(**run.config)

            model_name = measurements_distributions.ModelName[run.config['model_name'].split(".")[-1]]
            _model = measurements_distributions.get_measurement_distribution(
                model_name,
                d_x=run_parameters.d_x,
                d_p=run_parameters.d_p,
                norm_min=run_parameters.norm_min,
                norm_max=run_parameters.norm_max,
                a_limit=float(run_parameters.min_limit),
                b_limit=float(run_parameters.max_limit),
                non_linear_function=run_parameters.non_linear_function)
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
            return _model, run_parameters, _cnf.to(pru.get_working_device())
