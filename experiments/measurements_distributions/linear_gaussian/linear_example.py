import torch
import os
from experiments import constants
from experiments.measurements_distributions.base_model import BaseModel
from experiments.measurements_distributions.linear_gaussian.linear_optimal_flow import LinearOptimalFlow
from experiments.measurements_distributions import parameters_generator
import pyresearchutils as pru

VAR = 1.0


class LinearModel(BaseModel):
    def __init__(self, d_x: int, d_p: int, norm_min: float, norm_max: float, **kwargs):
        parameters = parameters_generator.ParameterContainer(
            parameters_generator.NormGaussian(constants.THETA, d_p, 0, VAR, norm_min, norm_max))
        super().__init__(d_x, d_p, parameters, has_optimal_flow=True, has_crb=True, has_mcrb=True)
        self.optimal_flow = LinearOptimalFlow(self.d_x, self.parameter_vector_length)

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

    def generate_data(self, n_samples, **kwargs):
        z = torch.randn([n_samples, self.d_x], device=pru.get_working_device())
        return self.optimal_flow.backward(z, **kwargs)[0]

    @staticmethod
    def ml_estimator(r):
        raise NotImplemented
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, *args, **kwargs):
        fim = torch.matmul(
            torch.matmul(self.optimal_flow.h.transpose(dim0=0, dim1=1), torch.linalg.inv(self.optimal_flow.c_xx)),
            self.optimal_flow.h)
        return torch.linalg.inv(fim)

    def mcrb(self, *args, **kwargs):
        raise NotImplemented
