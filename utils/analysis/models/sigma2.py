import numpy as np
import sys, os
import matplotlib.pyplot as plt
import gpmpcontrib.levelset.test_problems as test_problems

input_folder = sys.argv[1]
test_function = sys.argv[2]

seq_method = "straddle"

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}

n_runs = 30

idx_covparam = 0

problem = getattr(test_problems, test_function)

plt.figure()

for regp_method in ["None", "Constant", "Concentration", "Spatial"]:
    path = os.path.join(input_folder, seq_method, test_function, regp_method)

    full_data = []

    for i in range(n_runs):
        data = np.exp(np.load(os.path.join(path, "covparam_{}.npy".format(i))))
        data_i = data[:, idx_covparam]

        full_data.append(data_i)

    full_data = np.array(full_data)
    lower, med, upper = np.quantile(full_data, 0.1, axis=0), np.quantile(full_data, 0.5, axis=0), np.quantile(full_data, 0.9, axis=0)

    plt.fill_between(range(med.shape[0]), lower, upper, alpha=0.2, color=regp_methods_palette[regp_method])
    plt.plot(med, linestyle="solid", color=regp_methods_palette[regp_method])

plt.semilogy()
plt.show()