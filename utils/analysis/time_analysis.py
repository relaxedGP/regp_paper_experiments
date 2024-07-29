import numpy as np
import sys, os
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt
import gpmpcontrib.optim.test_problems as test_problems
import plotting_utils

input_folder = sys.argv[1]

test_function = sys.argv[2]

seq_strategy = sys.argv[3]

n_runs = int(sys.argv[4])

output_path = sys.argv[5]

# Get dim
problem = getattr(test_problems, test_function)
d = len(problem.input_box[0])

n0_over_d = 10
max_f_evals = 300

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}

regp_methods_times = {}
regp_methods_relative_times = {}

for k in regp_methods_palette.keys():
    res = []
    for i in range(n_runs):
        _res = np.load(os.path.join(input_folder, seq_strategy, test_function, k, "times_{}.npy".format(i)))
        res.append(_res)
    res = np.array(res)

    regp_methods_times[k] = [
        np.quantile(res, 0.1, axis=0),
        np.quantile(res, 0.5, axis=0),
        np.quantile(res, 0.9, axis=0)
    ]

    if k != "None":
        none_res = []
        for i in range(n_runs):
            _none_res = np.load(os.path.join(input_folder, seq_strategy, test_function, "None", "times_{}.npy".format(i)))
            none_res.append(_none_res)
        none_res = np.array(none_res)

    regp_methods_relative_times[k] = [
        np.quantile(res/none_res, 0.1, axis=0),
        np.quantile(res/none_res, 0.5, axis=0),
        np.quantile(res/none_res, 0.9, axis=0)
    ]

#
# plt.figure(figsize=(3.0, 2.6))
#
# plt.title("{}, {}".format(test_function, seq_strategy))
#
# for k in regp_methods_times.keys():
#     lower_q, med, upper_q = regp_methods_times[k]
#     plt.fill_between(range(lower_q.shape[0]), lower_q, upper_q, color=regp_methods_palette[k], alpha=0.2)
#     plt.plot(med, label=k, linestyle="solid", color=regp_methods_palette[k])
#
# plt.legend()
# plt.semilogy()
# plt.show()

#
plt.figure(figsize=(6 * 4/6, 4 * 4/6))

plt.title("{} (reGP relative running time)".format(plotting_utils.get_test_function_format(test_function)))

abscissa = list(range(n0_over_d * d - 1, max_f_evals))

for k in ["Concentration", "Constant", "Spatial"]:
    lower_q, med, upper_q = regp_methods_relative_times[k]

    plt.fill_between(abscissa, lower_q, upper_q, color=regp_methods_palette[k], alpha=0.2)
    plt.plot(abscissa, med, label=k, linestyle="solid", color=regp_methods_palette[k])

#plt.legend()
plt.ylim([1, 1000])
plt.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "{}.pgf".format(test_function)))
# plt.show()

#
# plt.figure()
#
# plt.title("{}, {}".format(test_function, seq_strategy))
#
# for k in ["Concentration", "Constant", "Spatial"]:
#     lower_q, med, upper_q = regp_methods_relative_times[k]
#
#     _, med_none, _ = regp_methods_times["None"]
#
#     plt.fill_between(med_none, lower_q, upper_q, color=regp_methods_palette[k], alpha=0.2)
#     plt.plot(med_none, med, label=k, linestyle="solid", color=regp_methods_palette[k])
#
# #plt.legend()
# plt.ylim([1, 1000])
# plt.semilogy()
# plt.tight_layout()
# plt.show()