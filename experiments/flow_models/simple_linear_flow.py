import normflowpy as nfp
import pyresearchutils as pru


def build_simple_linear_flow(in_prior, in_d_x, in_d_p, in_cond_name_list):
    input_vector_shape = [in_d_p]
    flows = []
    flows.append(
        nfp.flows.InvertibleFullyConnected(dim=in_d_x, random_initialization=True))
    flows.append(
        nfp.flows.AffineInjector(x_shape=input_vector_shape,
                                 cond_name_list=in_cond_name_list,
                                 condition_vector_size=in_d_p, n_hidden=-1,
                                 net_class=nfp.base_nets.generate_mlp_class(n_layer=1,
                                                                            non_linear_function=None),
                                 scale=False))
    return nfp.NormalizingFlowModel(in_prior, flows, condition_network=None).to(pru.torch.get_working_device())
