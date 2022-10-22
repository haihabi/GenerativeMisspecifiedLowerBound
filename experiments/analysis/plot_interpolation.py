import pickle
import numpy as np
from matplotlib import pyplot as plt
from experiments.analysis.analysis_tlinear_dataset import RUNS_DICT
import pyresearchutils as pru
import torch

mc_n = 100
norm_array = torch.linspace(0.1, 9, 20)

with open('../data/data_interpolation_reg_update_seed_2_trimming.pickle', 'rb') as handle:
    results_dict = pickle.load(handle)

for max_limit, r in results_dict.items():
    plt.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                 r[0],
                 "o",
                 label=r"$\overline{LB}$ " f"c={max_limit}")
    plt.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                 r[1],
                 "x",
                 label=r"GMLB " + f"c={max_limit}")
    plt.semilogy(pru.torch2numpy(norm_array.detach()),
                 r[2],
                 label=f"LB c={max_limit}")

ax = plt.gca()
axins = ax.inset_axes([0.58, 0.05, 0.4, 0.4])
for max_limit, r in results_dict.items():
    axins.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                   r[0],
                   "o",
                   label=r"$\overline{LB}$ " f"b=-a={max_limit}")
    axins.semilogy(pru.torch2numpy(norm_array.reshape([1, -1]).repeat([mc_n, 1])).flatten(),
                   r[1],
                   "x",
                   label=r"GMLB " + f"b=-a={max_limit}")
    axins.semilogy(pru.torch2numpy(norm_array.detach()),
                   r[2],
                   label=f"LB b=-a={max_limit}")
axins.set_xlim(1.8, 2.6)
axins.set_ylim(5.5, 10)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
plt.grid()
plt.legend()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\frac{\mathrm{Tr}(LB)}{d_p}$")
plt.tight_layout()
plt.savefig("compare_nltn.svg")
plt.show()
