import torch
from experiments.measurements_distributions.parameters_generator.base_parameter import BaseParameter
import pyresearchutils as pru


class CubicParameter(BaseParameter):
    def __init__(self, name: str, dim: int, min_value, max_value):
        super().__init__(name, dim)
        self.min_value = min_value * torch.ones([1, dim], device=pru.torch.get_working_device())
        self.max_value = max_value * torch.ones([1, dim], device=pru.torch.get_working_device())

    def parameter_range(self, n_steps, theta_scale_min=None, theta_scale_max=None):
        theta_min = self.min_value if theta_scale_min is None else theta_scale_min * self.min_value
        theta_max = self.max_value if theta_scale_max is None else theta_scale_max * self.max_value
        if torch.all(theta_min == theta_max):
            return theta_max * torch.ones([1, 1])
        return theta_min + (theta_max - theta_min) * torch.linspace(0, 1, n_steps,
                                                                    device=pru.torch.get_working_device()).reshape(
            [-1, 1])

    def random_sample(self):
        return self.min_value + (self.max_value - self.min_value) * torch.rand([1, self.dim],
                                                                               device=pru.torch.get_working_device())
