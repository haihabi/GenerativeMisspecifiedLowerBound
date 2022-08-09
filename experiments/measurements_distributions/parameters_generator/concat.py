from typing import List

import torch
from experiments.measurements_distributions.parameters_generator.base_parameter import BaseParameter


class ConcatParameters(BaseParameter):
    def __init__(self, name, *args):
        dim = sum([p.dim_real for p in args])
        super().__init__(name, dim, is_complex=False)
        self.param_list: List[BaseParameter] = args

    def random_sample(self):
        return torch.cat([p.random_sample() for p in self.param_list], dim=0)

    def parameter_range(self, n_steps, **kwargs):
        return torch.cat([p.parameter_range(n_steps, **kwargs) for p in self.param_list], dim=-1)
