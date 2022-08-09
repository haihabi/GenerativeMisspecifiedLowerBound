import torch
import normflowpy as nfp
from experiments import constants
from torch import nn


def generate_c_xx_matrix(dim):
    b = torch.randn([dim, dim], device=constants.DEVICE)
    b = b / torch.norm(b)
    return torch.matmul(b.transpose(dim0=0, dim1=1), b)


def generate_h_matrix(d_x, d_p):
    a = torch.randn([d_x, d_p], device=constants.DEVICE)
    return a / torch.sqrt(torch.pow(torch.abs(a), 2.0).sum())


class LinearOptimalFlow(nfp.ConditionalBaseFlowLayer):
    def __init__(self, dim, parameter_vector_size):
        super().__init__()
        self.h = nn.Parameter(generate_h_matrix(dim, parameter_vector_size), requires_grad=False)
        self.c_xx = nn.Parameter(generate_c_xx_matrix(dim), requires_grad=False)

        self.dim = dim
        l_matrix = torch.linalg.cholesky(self.c_xx)
        self.l_matrix = l_matrix
        self.l_matrix_inv = torch.linalg.inv(l_matrix)
        self.l_log_det = torch.log(torch.linalg.det(self.l_matrix))
        self.l_inv_log_det = torch.log(torch.linalg.det(self.l_matrix_inv))

    def forward(self, x, **kwargs):
        cond = kwargs[constants.THETA]
        z = torch.matmul(self.l_matrix_inv,
                         x.transpose(dim0=0, dim1=1) - torch.matmul(self.h, cond.transpose(dim0=0, dim1=1))).transpose(
            dim0=0, dim1=1)
        return z, self.l_inv_log_det

    def backward(self, z, **kwargs):
        cond = kwargs[constants.THETA]
        x = torch.matmul(self.l_matrix, z.transpose(dim0=0, dim1=1)) + torch.matmul(self.h, cond.transpose(dim0=0, dim1=1))
        x = x.transpose(dim0=0, dim1=1)
        return x, self.l_log_det
