import torch
import math
from experiments.measurements_distributions.parameters_generator.base_parameter import BaseParameter
import pyresearchutils as pru


def randcn(shape: tuple) -> torch.Tensor:
    """Samples from complex circularly-symmetric normal distribution.

    Args:
        shape (tuple): Shape of the output.

    Returns:
        ~numpy.ndarray: A complex :class:`~numpy.ndarray` containing the
        samples.
    """
    x = 1j * torch.randn(*shape)
    x += torch.randn(*shape)
    x *= math.sqrt(0.5)
    return x


class ComplexGaussianParameter(BaseParameter):
    def __init__(self, name: str, dim: int, mean_value, variance_value, n_std_range=6):
        super().__init__(name, dim, is_complex=True)
        self.mean_value = mean_value
        self.variance_value = variance_value
        self.n_std_range = n_std_range

    def parameter_range(self, n_steps, theta_scale_min=None, theta_scale_max=None):
        base_min_value = self.mean_value - self.n_std_range * math.sqrt(self.variance_value)
        base_max_value = self.mean_value + self.n_std_range * math.sqrt(self.variance_value)
        theta_min = base_min_value if theta_scale_min is None else theta_scale_min * base_min_value
        theta_max = base_max_value if theta_scale_max is None else theta_scale_max * base_max_value

        return theta_min + (theta_max - theta_min) * torch.linspace(0, 1, n_steps,
                                                                    device=pru.torch.get_working_device()).reshape(
            [-1, 1])

    def random_sample(self):
        x_complex = self.mean_value + self.variance_value * randcn((self.dim,))
        return torch.cat([torch.real(x_complex), torch.imag(x_complex)], dim=0)
