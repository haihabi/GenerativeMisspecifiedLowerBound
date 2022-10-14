from experiments.flow_models.simple_linear_flow import build_simple_linear_flow
from experiments.flow_models.prior_builder import gaussian_prior_builder


def generate_cnf_model(in_d_x, in_d_p, in_cond_name_list, n_blocks=1,
                       n_layer_inject=1, n_hidden_inject=-1, inject_scale=False, inject_bias=False,
                       affine_inject=False):
    prior = gaussian_prior_builder(in_d_x)
    return build_simple_linear_flow(prior, in_d_x, in_d_p, in_cond_name_list, n_blocks=n_blocks,
                                    n_layer_inject=n_layer_inject, n_hidden_inject=n_hidden_inject,
                                    inject_scale=inject_scale, inject_bias=inject_bias, affine_inject=affine_inject)
