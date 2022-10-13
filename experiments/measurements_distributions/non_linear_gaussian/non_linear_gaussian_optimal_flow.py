import torch
import normflowpy as nfp
from experiments import constants
from torch import nn
from experiments.measurements_distributions.linear_gaussian.linear_optimal_flow import generate_c_xx_matrix, \
    generate_h_matrix
from experiments.measurements_distributions.linear_truncated_gaussian.softclip import soft_clip


class NonLinearOptimalFlow(nfp.ConditionalBaseFlowLayer):
    def __init__(self, dim, parameter_vector_size, a_limit: float = 5, b_limit: float = 5):
        super().__init__()
        self.h = nn.Parameter(generate_h_matrix(dim, parameter_vector_size), requires_grad=False)
        self.c_xx = nn.Parameter(generate_c_xx_matrix(dim), requires_grad=False)
        self.a_limit = nn.Parameter(a_limit * torch.ones(1), requires_grad=False)
        self.b_limit = nn.Parameter(b_limit * torch.ones(1), requires_grad=False)
        self.dim = dim
        self.build_sub_parameters()

    def build_sub_parameters(self):
        l_matrix = torch.linalg.cholesky(self.c_xx)
        self.l_matrix = l_matrix
        self.l_matrix_inv = torch.linalg.inv(l_matrix)
        self.l_log_det = torch.log(torch.linalg.det(self.l_matrix))
        self.l_inv_log_det = torch.log(torch.linalg.det(self.l_matrix_inv))

    def forward(self, x, **kwargs):
        cond = kwargs[constants.THETA]
        mu = torch.matmul(self.h, cond.transpose(dim0=0, dim1=1))
        mu = soft_clip(mu, torch.min(self.a_limit), torch.max(self.b_limit))
        mu = torch.clip(mu, min=torch.min(self.a_limit), max=torch.max(self.b_limit))
        z = torch.matmul(self.l_matrix_inv,
                         x.transpose(dim0=0, dim1=1) - mu).transpose(
            dim0=0, dim1=1)
        return z, self.l_inv_log_det

    def backward(self, z, **kwargs):
        cond = kwargs[constants.THETA]
        mu = torch.matmul(self.h, cond.transpose(dim0=0, dim1=1))
        mu = soft_clip(mu, torch.min(self.a_limit), torch.max(self.b_limit))
        mu = torch.clip(mu, min=torch.min(self.a_limit), max=torch.max(self.b_limit))
        x = torch.matmul(self.l_matrix, z.transpose(dim0=0, dim1=1)) + mu
        x = x.transpose(dim0=0, dim1=1)
        return x, self.l_log_det
