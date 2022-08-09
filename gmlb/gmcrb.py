import normflowpy as nfp
from gmlb.misspecified_model import BaseMisSpecifiedModel
from gmlb.emcrb import estimate_mcrb, compute_lower_bound


def generative_misspecified_cramer_rao_bound(generative_model: nfp.NormalizingFlowModel, m,
                                             ms_model: BaseMisSpecifiedModel,
                                             parameter_name="theta", **kwargs):
    theta_true = kwargs.get(parameter_name)
    x_s = generative_model.sample(m, **kwargs)
    p_zero = ms_model.mml(x_s)
    mcrb = estimate_mcrb(x_s, p_zero, ms_model)
    lb = compute_lower_bound(mcrb, theta_true, p_zero)
    return mcrb, lb
