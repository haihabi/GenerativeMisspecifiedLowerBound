import torch

from experiments.analysis.helpers import create_model_delta, get_h_and_c_xx, build_misspecifietion_type_one, \
    parameter_sweep
from experiments.measurements_distributions import TruncatedLinearModel
from matplotlib import pyplot as plt

from experiments.measurements_distributions.linear_truncated_gaussian.softclip import soft_clip

if __name__ == '__main__':
    d_x = d_p = 8
    norm_max = 10
    n_test = 20
    m = 64000
    alpha = beta = 0.1
    b = 5.0
    a = -b

    model = TruncatedLinearModel(d_x, d_p, 0.1, 10, a_limit=a, b_limit=b)
    h_delta, l_delta = create_model_delta(d_x, d_p)
    h, c_xx_overline = get_h_and_c_xx(model)
    l_x = torch.linalg.cholesky(c_xx_overline)
    linear_ms = build_misspecifietion_type_one(h, l_x, h_delta, l_delta, alpha, beta)
    p_true = model.parameters.random_sample_parameters()
    _, _, lb_array_z, norm_array = parameter_sweep(None, p_true, n_test, linear_ms, m,
                                                   model,
                                                   norm_max=norm_max,
                                                   run_optimal=False,
                                                   run_model=False,
                                                   non_linear=True)
    plt.subplot(1, 2, 1)
    plt.semilogy(norm_array.detach().numpy(),
                 (torch.diagonal(lb_array_z, dim1=1, dim2=2).sum(dim=-1) / d_p).detach().numpy(),
                 label=f"LB")
    plt.subplot(1, 2, 2)
    x = torch.linspace(-10, 10, 1000)
    plt.plot(x.numpy(), soft_clip(x, a, b))
    plt.show()
