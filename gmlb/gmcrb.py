import normflowpy as nfp
from gmlb.misspecified_model import BaseMisSpecifiedModel
from gmlb.emcrb import estimate_mcrb, compute_lower_bound


def generative_misspecified_cramer_rao_bound(data_generator, m,
                                             ms_model: BaseMisSpecifiedModel,
                                             parameter_name="theta", **kwargs):
    x_s = data_generator(m, **kwargs)
    theta_true = kwargs.get(parameter_name)
    p_zero = ms_model.mml(x_s)
    mcrb = estimate_mcrb(x_s, p_zero, ms_model)
    lb = compute_lower_bound(mcrb, theta_true, p_zero)
    return mcrb, lb, p_zero


def generative_misspecified_cramer_rao_bound_flow(generative_model: nfp.NormalizingFlowModel, m,
                                                  ms_model: BaseMisSpecifiedModel,
                                                  parameter_name="theta", **kwargs):
    def _data_gen(in_m, **in_kwargs):
        return generative_model.sample(in_m, **in_kwargs)

    return generative_misspecified_cramer_rao_bound(_data_gen, m, ms_model, parameter_name, **kwargs)
