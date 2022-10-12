import torch
import os
from experiments import constants
from experiments.measurements_distributions.base_model import BaseModel
from experiments.measurements_distributions.non_linear_gaussian.non_linear_gaussian_optimal_flow import \
    NonLinearOptimalFlow
from experiments.measurements_distributions import parameters_generator
import pyresearchutils as pru

VAR = 1.0


class NonLinearGaussian(BaseModel):
    def __init__(self, d_x: int, d_p: int, norm_min: float, norm_max: float, a_limit: float = 5, b_limit: float = 5,
                 **kwargs):
        parameters = parameters_generator.ParameterContainer(
            parameters_generator.NormGaussian(constants.THETA, d_p, 0, VAR, norm_min, norm_max))
        super().__init__(d_x, d_p, parameters, has_optimal_flow=True, has_crb=False, has_mcrb=True)
        self.optimal_flow = NonLinearOptimalFlow(self.d_x, self.parameter_vector_length, a_limit, b_limit)
        self.a_limit = a_limit
        self.b_limit = b_limit

    @property
    def h(self):
        return self.optimal_flow.h

    @property
    def c_xx_bar(self):
        return self.optimal_flow.c_xx

    def _get_optimal_model(self):
        return self.optimal_flow

    @property
    def file_name(self):
        return f"{self.name}_model.pt"

    def save_data_model(self, folder):
        torch.save(self.optimal_flow.state_dict(), os.path.join(folder, self.file_name))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, self.file_name), map_location="cpu")
        self.optimal_flow.load_state_dict(data)
        self.optimal_flow.build_sub_parameters()

    def generate_data(self, n_samples, **kwargs):
        z = torch.randn([n_samples, self.d_x], device=pru.get_working_device())
        return self.optimal_flow.backward(z, **kwargs)[0]
