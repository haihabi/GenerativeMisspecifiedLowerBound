import torch
import os
from experiments import constants
from experiments.measurements_distributions.base_model import BaseModel
from experiments.measurements_distributions.linear_gaussian.linear_optimal_flow import LinearOptimalFlow
from experiments.measurements_distributions import parameters_generator

VAR = 1.0


class LinearModel(BaseModel):
    def __init__(self, d_x: int, d_p: int, norm_min: float, norm_max: float):
        parameters = parameters_generator.ParameterContainer(
            parameters_generator.NormGaussian(constants.THETA, d_p, 0, VAR, norm_min, norm_max))
        super().__init__(d_x, parameters, has_optimal_flow=True, has_crb=True, has_mcrb=True)
        self.optimal_flow = LinearOptimalFlow(self.d_x, self.parameter_vector_length)

    def _get_optimal_model(self):
        return self.optimal_flow

    def save_data_model(self, folder):
        torch.save(self.optimal_flow.state_dict(), os.path.join(folder, f"{self.name}_model.pt"))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, f"{self.name}_model.pt"), map_location="cpu")
        self.optimal_flow.load_state_dict(data)

    def generate_data(self, n_samples, **kwargs):
        z = torch.randn([n_samples, self.d_x], device=constants.DEVICE)
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

# if __name__ == '__main__':
#     import gcrb
#     import numpy as np
#
#     dm = LinearModel(10, 2, -10, 10, 0.1)
#     theta_array = dm.parameter_range(5)
#     model_opt = dm.get_optimal_model()
#     crb_list = [dm.crb(theta) for theta in theta_array]
#     gcrb_list = [torch.inverse(gcrb.adaptive_sampling_gfim(model_opt, theta.reshape([-1]))) for theta in theta_array]
#
#     theta_array = theta_array.cpu().detach().numpy()
#     crb_array = torch.stack(crb_list).cpu().detach().numpy()
#     gcrb_array = torch.stack(gcrb_list).cpu().detach().numpy()
#     from matplotlib import pyplot as plt
#
#     plt.plot(theta_array[:, 0], np.diagonal(crb_array, axis1=1, axis2=2).sum(axis=-1))
#     plt.plot(theta_array[:, 0], np.diagonal(gcrb_array, axis1=1, axis2=2).sum(axis=-1))
#     plt.show()
# print("a")
