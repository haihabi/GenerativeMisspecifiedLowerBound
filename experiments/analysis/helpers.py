import torch
from experiments.mcrb.linear_mcrb import LinearMCRB


def build_misspecifietion_type_one(in_h, in_l_xx, in_h_delta, in_l_vv_delta, in_alpha, in_beta):
    h_matrix = in_h + in_alpha * in_h_delta
    l_c = in_l_xx + in_beta * in_l_vv_delta
    c_ww_matrix = torch.matmul(l_c, l_c.T)
    return LinearMCRB(h_matrix, c_ww_matrix)
