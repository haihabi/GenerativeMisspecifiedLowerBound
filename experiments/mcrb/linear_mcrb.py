import torch
import gmlb


class LinearMCRB(gmlb.BaseMisSpecifiedModel):
    def __init__(self, h, c_vv):
        self.h = h
        self.p_dim = h.shape[-1]
        self.c_vv = c_vv
        self.c_vv_inv = torch.linalg.inv(self.c_vv)
        self._fim = torch.matmul(torch.matmul(self.h.T, self.c_vv_inv), self.h)
        self._crb = torch.linalg.inv(self._fim)

    # TODO: remove H
    def calculate_mcrb(self, h, c_vv):
        a_inv = self.crb()
        b = torch.matmul(torch.matmul(torch.matmul(self.h.T, self.c_vv_inv), c_vv), torch.matmul(self.c_vv_inv, self.h))
        return torch.matmul(torch.matmul(a_inv, b), a_inv)

    # TODO: change to mu
    def calculate_pseudo_true_parameter(self,in_mu):
        u = torch.matmul(self.h.T, torch.matmul(self.c_vv_inv, in_mu))
        return torch.matmul(self.crb(), u)

    def crb(self):
        return self._crb

    def mml(self, x):
        x_mean = torch.mean(x, dim=0)
        return torch.matmul(self.crb(), torch.matmul(self.h.T, torch.matmul(self.c_vv_inv, x_mean)))

    def log_likelihood_hessian(self, x, p):
        return self._fim.reshape([1, self.p_dim, self.p_dim])

    def log_likelihood_jacobian(self, x, p):
        return torch.matmul(self.h.T, torch.matmul(self.c_vv_inv, (x - torch.matmul(self.h, p).reshape([1, -1])).T)).T
