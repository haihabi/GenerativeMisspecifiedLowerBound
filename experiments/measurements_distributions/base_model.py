import os
import pyresearchutils as pru
import normflowpy as nf
import torch
from tqdm import tqdm
from experiments import constants
from experiments.measurements_distributions import parameters_generator
from torch.distributions import MultivariateNormal


class BaseModel(object):
    def __init__(self, d_x, parameters: parameters_generator.ParameterContainer, has_optimal_flow=False, has_crb=True,
                 has_mcrb=True):
        self.d_x = d_x
        self.parameters = parameters
        self.has_crb = has_crb
        self.has_mcrb = has_mcrb
        self.has_optimal_flow = has_optimal_flow
        if self.parameters.get_parameter(constants.THETA) is None:
            pru.logger.error("Theta don\'t exist in parameter container")

    def generate_data(self, n_samples, **kwargs):
        raise NotImplemented

    def crb(self, *args, **kwargs):
        raise NotImplemented

    def _get_optimal_model(self):
        raise NotImplemented

    def get_optimal_model(self):
        prior = MultivariateNormal(torch.zeros(self.d_x, device=pru.torch.get_working_device()),
                                   torch.eye(self.d_x, device=pru.torch.get_working_device()))
        return nf.NormalizingFlowModel(prior, [self._get_optimal_model()]).to(pru.torch.get_working_device())

    def model_exist(self, folder):
        return os.path.isfile(os.path.join(folder, f"{self.name}_model.pt"))

    @property
    def parameter_vector_length(self):
        return self.parameters.get_parameter(constants.THETA).dim

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{self.d_x}_{self.parameter_vector_length}"

    def parameter_range(self, n_steps, **kwargs):
        return self.parameters.parameter_range(n_steps, **kwargs)

    def build_dataset(self, dataset_size, transform):
        data = []
        label = []
        for _ in tqdm(range(dataset_size)):
            param_dict = self.parameters.random_sample_parameters()
            signal = self.generate_data(1, **param_dict)
            data.append(signal.detach().cpu().numpy().flatten())
            label.append({k: v.detach().cpu().numpy().flatten() for k, v in param_dict.items()})

        return pru.NumpyDataset(data, label, transform)
