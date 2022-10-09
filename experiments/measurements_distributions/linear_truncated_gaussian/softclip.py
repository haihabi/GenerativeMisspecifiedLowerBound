import torch
import numpy as np

SLOP = 1.0


def soft_clip(x, a, b, r=0.5):
    # z = x.clone()
    # c = (b - a)
    # z[x > b - r] = b - r + r * (1 - torch.exp(-torch.abs(x[x > b - r] - b + r)))
    # z[x < a + r] = a + r - r * (1 - torch.exp(-torch.abs(x[x < a + r] - a - r)))
    # z = (b - a) * torch.sigmoid(x) + a
    # z = (x - a) * torch.sigmoid(x-a)
    # s = a
    c = b - a
    z = c * torch.arctan(x) / np.pi
    return z


if __name__ == '__main__':
    x = torch.linspace(-10, 10, 10000)
    y = soft_clip(x, -2, 5, 0.5)
    from matplotlib import pyplot as plt

    plt.plot(x.numpy(), y.numpy())
    plt.plot(x.numpy(), x.numpy())
    plt.grid()
    plt.show()
