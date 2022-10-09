import normflowpy as nfp
from gmlb.misspecified_model import BaseMisSpecifiedModel
from gmlb.emcrb import estimate_mcrb, compute_lower_bound
import torch


def trim(in_data, in_min_limit, in_max_limit):
    if in_min_limit is not None:
        low_state = torch.all(in_data > in_min_limit.reshape([1, -1]), dim=-1)
        in_data = in_data[low_state, :]
    if in_max_limit is not None:
        high_state = torch.all(in_data < in_max_limit.reshape([1, -1]), dim=-1)
        in_data = in_data[high_state, :]
    return in_data


def generative_misspecified_cramer_rao_bound(data_generator, m,
                                             ms_model: BaseMisSpecifiedModel,
                                             parameter_name="theta", min_limit=None, max_limit=None, **kwargs):
    with torch.no_grad():
        x_s = data_generator(m, **kwargs)
        if min_limit is not None or max_limit is not None:
            x_s_trim = trim(x_s, min_limit, max_limit)
            while x_s_trim.shape[0] < m:
                x_s = data_generator(m, **kwargs)
                x_s_trim = torch.cat([x_s_trim, trim(x_s, min_limit, max_limit)])
            x_s = x_s_trim[:m, :]

        theta_true = kwargs.get(parameter_name)
        p_zero = ms_model.mml(x_s)
        mcrb, a_matrix, b_matrix = estimate_mcrb(x_s, p_zero, ms_model)
        lb = compute_lower_bound(mcrb, theta_true, p_zero)

    return mcrb, lb, p_zero


def generative_misspecified_cramer_rao_bound_flow(generative_model: nfp.NormalizingFlowModel, m,
                                                  ms_model: BaseMisSpecifiedModel,
                                                  parameter_name="theta", min_limit=None, max_limit=None, **kwargs):
    def _data_gen(in_m, **in_kwargs):
        return generative_model.sample(in_m, **in_kwargs)

    return generative_misspecified_cramer_rao_bound(_data_gen, m, ms_model, parameter_name, min_limit=min_limit,
                                                    max_limit=max_limit, **kwargs)
