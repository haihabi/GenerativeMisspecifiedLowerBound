import torch
import pyresearchutils as pru
from torch import distributions


def gaussian_prior_builder(size):
    prior = distributions.MultivariateNormal(torch.zeros(size, device=pru.torch.get_working_device()),
                                             torch.eye(size, device=pru.torch.get_working_device()))

    return prior
