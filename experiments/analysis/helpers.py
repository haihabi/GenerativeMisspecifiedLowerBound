import torch
import wandb
import os
from experiments import constants
from experiments.mcrb.linear_mcrb import LinearMCRB
from experiments import measurements_distributions
from experiments import flow_models
from argparse import Namespace

FLOW_BEST = "flow_best.pt"


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
                norm_max=run_parameters.norm_max)
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
