from experiments.flow_models.simple_linear_flow import build_simple_linear_flow
from experiments.flow_models.prior_builder import gaussian_prior_builder


def generate_cnf_model(in_d_x, in_d_p, in_cond_name_list):
    prior = gaussian_prior_builder(in_d_x)
    return build_simple_linear_flow(prior, in_d_x, in_d_p, in_cond_name_list)
