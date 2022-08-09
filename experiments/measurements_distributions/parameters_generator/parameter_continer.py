from typing import Dict
from experiments import constants
import pyresearchutils as pru
from experiments.measurements_distributions.parameters_generator.base_parameter import BaseParameter


class ParameterContainer:
    def __init__(self, *args):

        if len(set([p.name for p in args])) != len(args):
            pru.logger.error("All parameters must be unique")
        if not any([p.name == constants.THETA for p in args]):
            pru.logger.error("Theta parameter must be exists")
        self.parameters_dict: Dict[str, BaseParameter] = {p.name: p for p in args}

    def get_parameter(self, name: str) -> BaseParameter:
        return self.parameters_dict.get(name)

    def random_sample_parameters(self) -> dict:
        return {k: v.random_sample() for k, v in self.parameters_dict.items()}

    def parameter_range(self, n_steps, **kwargs):
        return {p.name: p.parameter_range(n_steps, **kwargs) for p in self.parameters_dict.values()}
