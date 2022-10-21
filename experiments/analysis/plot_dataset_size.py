import pickle
import numpy as np
from matplotlib import pyplot as plt
from experiments.analysis.analysis_tlinear_dataset import RUNS_DICT

with open('../data/data_reg.pickle', 'rb') as handle:
    results_dict = pickle.load(handle)

    # pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
fig, ax1 = plt.subplots(1, 1)
ax1.set_xscale("log")
ax1.set_yscale("log")
for a, r in results_dict.items():
    # if a == 3:
    #     continue
    r = np.asarray(r)
    ax1.errorbar(list(RUNS_DICT[a].keys()), r[:, 0], fmt="--",
                 label=r"$\overline{\mathrm{LB}}$ $c$=" + f"{a}")
    ax1.errorbar(list(RUNS_DICT[a].keys()), r[:, 1], label=r"$\mathrm{GMLB}$ $c$=" + f"{a}")

plt.legend()
plt.xlabel("Dataset-size")
plt.ylabel("xRE")
plt.grid()
plt.tight_layout()
plt.savefig("dataset-size-effect-nl.svg")
plt.show()
