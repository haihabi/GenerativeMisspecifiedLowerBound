import torch
import numpy as np

SLOP = 0.3


def soft_clip(x, a, b):
    # c = b - a
    # z = c * torch.arctan(SLOP * x) / np.pi
    z = 12 * torch.arctan(SLOP * x) / np.pi
    return z
