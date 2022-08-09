import torch
from gmlb.misspecified_model import BaseMisSpecifiedModel


def estimate_mcrb(x: torch.Tensor, p_zero: torch.Tensor, ms_model: BaseMisSpecifiedModel):
    a_matrix = torch.mean(ms_model.log_likelihood_hessian(x, p_zero), dim=0)
    inv_a_matrix = torch.linalg.inv(a_matrix)
    score = ms_model.log_likelihood_jacobian(x, p_zero)
    b_matrix = torch.mean(
        torch.matmul(score.unsqueeze(dim=-1), score.unsqueeze(dim=1)), dim=0)
    mcrb = torch.matmul(torch.matmul(inv_a_matrix, b_matrix), inv_a_matrix)
    return mcrb


def compute_lower_bound(in_mcrb, theta_true, theta_zero):
    r = theta_true - theta_zero
    lb = in_mcrb + torch.matmul(r, r.T)
    return lb
