import pickle
import numpy as np
from matplotlib import pyplot as plt
from experiments.analysis.analysis_tlinear_dataset import RUNS_DICT

with open('../data/data_reg_100_update_v2_seed.pickle', 'rb') as handle:
    results_dict = pickle.load(handle)

with open('../data/data_reg_100_update_v2_seed_trimming.pickle', 'rb') as handle:
    results_dict_trimming = pickle.load(handle)

a3 = 0.0055
a5 = 0.0060
# pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
fig, ax1 = plt.subplots(1, 1)
ax1.set_xscale("log")
ax1.set_yscale("log")
# for a, r in results_dict.items():
#     # if a == 3:
#     #     continue
#     # r = np.asarray(r)
#     r_array = np.asarray(list(r.values()))
#     # ax1.errorbar(list(RUNS_DICT[a].keys()), r_array[:, 0], fmt="--",
#     #              label=r"$\overline{\mathrm{LB}}$ $c$=" + f"{a}")
#     ax1.plot(list(RUNS_DICT[a].keys()), r_array[:, 1], label=r"$\mathrm{GMLB}$ $c$=" + f"{a}")

for a, r in results_dict_trimming.items():
    # if a == 3:
    #     continue
    # r = np.asarray(r)
    r_array = np.asarray(list(r.values()))
    ax1.plot(list(RUNS_DICT[a].keys()), r_array[:, 0], "--",
             label=r"$\overline{\mathrm{LB}}$ $c$=" + f"{a}")
    ax1.plot(list(RUNS_DICT[a].keys()), r_array[:, 1], label=r"Trimming $\mathrm{GMLB}$ $c$=" + f"{a}")

plt.legend()
plt.xlabel("Dataset-size")
plt.ylabel("xRE")
plt.grid()
plt.tight_layout()
plt.savefig("dataset-size-effect-nl.svg")
plt.show()
