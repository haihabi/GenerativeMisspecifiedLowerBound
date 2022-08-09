import torch


class BaseMisSpecifiedModel(object):

    def calculate_mcrb(self, *args, **kwargs):
        raise NotImplemented

    def mml(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def log_likelihood_hessian(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def log_likelihood_jacobian(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        raise NotImplemented
