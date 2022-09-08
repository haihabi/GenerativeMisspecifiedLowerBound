from enum import Enum
from experiments.measurements_distributions.linear_truncated_gaussian.linear_truncated import TruncatedLinearModel
from experiments.measurements_distributions.linear_gaussian.linear_example import LinearModel
import pyresearchutils as pru


class ModelName(Enum):
    LinearGaussian = 0
    LinearTruncatedGaussian = 1


model_dict = {ModelName.LinearTruncatedGaussian: TruncatedLinearModel,
              ModelName.LinearGaussian: LinearModel}


def get_measurement_distribution(name: ModelName, **kwargs):
    model_class = model_dict.get(name)
    if model_class is None:
        pru.logger.critical(f"Model Name:{name} not found")
    return model_class(**kwargs)
    # pass
