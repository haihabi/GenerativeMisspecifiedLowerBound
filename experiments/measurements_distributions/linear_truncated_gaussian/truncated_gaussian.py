import numpy as np
import torch
import pyresearchutils as pru

SQRT2 = np.sqrt(2)
SQRT2PI = SQRT2 * np.sqrt(np.pi)


def pdf(in_x):
    return torch.exp(-0.5 * torch.pow(in_x, 2.0)) / SQRT2PI


def cdf(in_x):
    return 0.5 * (1 + torch.erf(in_x / SQRT2))


def inv_cdf(in_x, eps=1e-8):
    return torch.erfinv(torch.clamp(2 * in_x - 1, -1 + eps, 1.0 - eps)) * SQRT2


def compute_truncted_normal_parameters(in_a, in_b, in_mu, in_sigma):
    alpha = (in_a - in_mu) / in_sigma
    beta = (in_b - in_mu) / in_sigma
    alpha_cdf = cdf(alpha)
    z = cdf(beta) - alpha_cdf
    return z, alpha, beta, alpha_cdf


def pdf_truncted_normal(in_x, in_a, in_b, in_mu, in_sigma):
    z, _, _, _ = compute_truncted_normal_parameters(in_a, in_b, in_mu, in_sigma)
    eps = (in_x - in_mu) / in_sigma
    return pdf(eps) / (z * in_sigma)
