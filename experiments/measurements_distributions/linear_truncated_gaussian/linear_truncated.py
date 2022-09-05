import torch
import os
from experiments import constants
from experiments.measurements_distributions.base_model import BaseModel
from experiments.measurements_distributions.linear_gaussian.linear_optimal_flow import generate_c_xx_matrix, \
    generate_h_matrix
from experiments.measurements_distributions import parameters_generator
from experiments.measurements_distributions.linear_truncated_gaussian.truncated_gaussian import sample_truncated_normal
import pyresearchutils as pru

VAR = 1.0


class TruncatedLinearModel(BaseModel):
    def __init__(self, d_x: int, d_p: int, norm_min: float, norm_max: float, a_limit: float = 5, b_limit: float = 5):
        parameters = parameters_generator.ParameterContainer(
            parameters_generator.NormGaussian(constants.THETA, d_p, 0, VAR, norm_min, norm_max))
        super().__init__(d_x, parameters, has_optimal_flow=True, has_crb=False, has_mcrb=True)
        self.h = generate_h_matrix(d_x, d_p)
        self.c_xx_bar = torch.diag(generate_c_xx_matrix(d_x).diag())
        self.a_limit = a_limit
        self.b_limit = b_limit
        self.a = -a_limit * torch.rand([d_x])
        self.b = b_limit * torch.rand([d_x])

    @property
    def name(self) -> str:
        return f"{super(TruncatedLinearModel, self).name}_{self.a_limit}_{self.b_limit}"

    def state_dict(self):
        return {"h": self.h,
                "c_xx_bar": self.c_xx_bar,
                "a": self.a,
                "b": self.b}

    @property
    def file_name(self):
        return f"{self.name}_model.pt"

    def save_data_model(self, folder):
        torch.save(self.state_dict(), os.path.join(folder, self.file_name))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, self.file_name), map_location="cpu")
        raise NotImplemented

    def generate_data(self, n_samples, **kwargs):
        mu = torch.matmul(kwargs[constants.THETA], self.h.T)
        x_s = sample_truncated_normal([n_samples, self.d_x], self.a, self.b, mu, torch.sqrt(self.c_xx_bar.diag()))
        return x_s

    @staticmethod
    def ml_estimator(r):
        raise NotImplemented

    def mcrb(self, *args, **kwargs):
        raise NotImplemented
