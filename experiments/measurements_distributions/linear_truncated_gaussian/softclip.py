import torch
import numpy as np

SLOP = 3.3


def soft_clip(x, a, b):
    c = (b - a)
    z = torch.arctan(SLOP * (x - a) / c) / np.pi + 0.5  # [0,1]
    return c * z + a
