import normflowpy as nfp
import pyresearchutils as pru


def build_simple_linear_flow(in_prior, in_d_x, in_d_p, in_cond_name_list, n_blocks=1,
                             n_layer_inject=1, n_hidden_inject=-1, inject_scale=False, inject_bias=False):
    pru.logger.info(f"Run CNF with parameter:{locals()}")
    input_vector_shape = [in_d_x]
    flows = []
    if n_blocks < 1:
        pru.logger.critical("A minimum of a single block is required")
    for b in range(n_blocks):
        flows.append(
            nfp.flows.ActNorm(input_vector_shape))
        flows.append(
            nfp.flows.InvertibleFullyConnected(dim=in_d_x, random_initialization=True))

        flows.append(
            nfp.flows.AffineInjector(x_shape=input_vector_shape,
                                     cond_name_list=in_cond_name_list,
                                     condition_vector_size=in_d_p, n_hidden=n_hidden_inject,
                                     net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_inject,
                                                                                bias=inject_bias),
                                     scale=inject_scale))
        # if coupling_layer and b != (n_blocks - 1):
        #     flows.append(nfp.flows.AffineCoupling())
    return nfp.NormalizingFlowModel(in_prior, flows, condition_network=None).to(pru.torch.get_working_device())
