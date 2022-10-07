import torch
import os
import numpy as np
from experiments import constants
from experiments.measurements_distributions.base_model import BaseModel
from experiments.measurements_distributions.linear_gaussian.linear_optimal_flow import generate_c_xx_matrix, \
    generate_h_matrix
from experiments.measurements_distributions import parameters_generator
from experiments.measurements_distributions.linear_truncated_gaussian.minimax_tilting_sampler import TruncatedMVN
import pyresearchutils as pru
from experiments.measurements_distributions.linear_truncated_gaussian.softclip import soft_clip

VAR = 1.0


class TruncatedLinearModel(BaseModel):
    def __init__(self, d_x: int, d_p: int, norm_min: float, norm_max: float, a_limit: float = 5, b_limit: float = 5,
                 non_linear_function=True):
        parameters = parameters_generator.ParameterContainer(
            parameters_generator.NormGaussian(constants.THETA, d_p, 0, VAR, norm_min, norm_max))
        super().__init__(d_x, d_p, parameters, has_optimal_flow=True, has_crb=False, has_mcrb=True)
        self.h = generate_h_matrix(d_x, d_p)
        self.c_xx_bar = generate_c_xx_matrix(d_x).diag()
        self.a_limit = a_limit
        self.b_limit = b_limit
        if self.a_limit > self.b_limit:
            pru.logger.critical(
                f"A limit must be smaller than b limit the given values are a={self.a_limit} and b={self.b_limit}")
        self.delta = b_limit - a_limit

        self.a = a_limit * torch.ones([d_x])
        self.b = b_limit * torch.ones([d_x])
        self.non_linear_function = non_linear_function

    @property
    def name(self) -> str:
        if self.non_linear_function:
            return f"{super(TruncatedLinearModel, self).name}_{self.a_limit}_{self.b_limit}"
        else:
            return f"{super(TruncatedLinearModel, self).name}_{self.a_limit}_{self.b_limit}_identity"

    def state_dict(self):
        return {"h": self.h,
                "c_xx_bar": self.c_xx_bar,
                "a": self.a,
                "b": self.b,
                "non_linear_function": self.non_linear_function}

    @property
    def file_name(self):
        return f"{self.name}_model.pt"

    def save_data_model(self, folder):
        torch.save(self.state_dict(), os.path.join(folder, self.file_name))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, self.file_name), map_location="cpu")
        self.h = data["h"]
        self.c_xx_bar = data["c_xx_bar"]
        self.a = data["a"]
        self.b = data["b"]
        self.non_linear_function = data["non_linear_function"]

    def generate_data(self, n_samples, **kwargs):
        mu = torch.matmul(kwargs[constants.THETA], self.h.T)
        if self.non_linear_function:
            mu = soft_clip(mu, torch.min(self.a), torch.max(self.b))
        mu = torch.clip(mu, min=torch.min(self.a), max=torch.max(self.b))
        tmvn = TruncatedMVN(pru.torch2numpy(mu).flatten(),
                            pru.torch2numpy(self.c_xx_bar.clone()),
                            pru.torch2numpy(self.a),
                            pru.torch2numpy(self.b))
        if n_samples > 1:
            if mu.shape[0] == 1:
                x_s = tmvn.sample(n_samples).T
            else:
                raise NotImplemented

        else:
            x_s = tmvn.sample(n_samples).T
        if not tmvn.success:
            return None
        return pru.change2torch(x_s)

    @staticmethod
    def ml_estimator(r):
        raise NotImplemented
