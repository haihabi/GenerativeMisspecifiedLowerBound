import torch
from experiments.measurements_distributions.parameters_generator.base_parameter import BaseParameter
import pyresearchutils as pru


class NormGaussian(BaseParameter):
    def __init__(self, name: str, dim: int, mean_value, variance_value, min_norm, max_norm):
        super().__init__(name, dim)
        self.mean_value = mean_value
        self.variance_value = variance_value
        self.min_norm = min_norm
        self.max_norm = max_norm
        if self.min_norm > self.max_norm:
            pru.logger.error("Minimal norm value must be smaller than maximal norm value")

    def generate_single_norm(self):
        z = torch.randn([1, self.dim],
                        device=pru.torch.get_working_device())
        return z / torch.linalg.norm(z, dim=- 1)

    def parameter_range(self, n_steps, theta_scale_min=None, theta_scale_max=None):
        z_base = self.generate_single_norm()
        min_norm = self.min_norm if theta_scale_min is None else theta_scale_min * self.min_norm
        max_norm = self.max_norm if theta_scale_max is None else theta_scale_max * self.max_norm
        if torch.all(min_norm == max_norm):
            norm = max_norm * torch.ones([1, 1])
        else:
            norm = min_norm + (max_norm - min_norm) * torch.linspace(0, 1, n_steps,
                                                                     device=pru.torch.get_working_device()).reshape(
                [-1, 1])
        return z_base * norm

    def random_sample(self):
        z = self.generate_single_norm()
        norm = self.min_norm + (self.max_norm - self.min_norm) * torch.rand([1, self.dim],
                                                                            device=pru.torch.get_working_device())
        return norm * z
