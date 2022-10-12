import torch
import numpy as np

SLOP = 0.1


def soft_clip(x, a, b):
    c = b - a
    z = c * torch.arctan(SLOP * x) / np.pi
    return z
