import sys, os
from plotting_utils import plot_noisy_optim, plot_value_of_estimated_minimizer, plot_error_on_estimated_minimizer

sequential_strategy = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]
n_runs = int(sys.argv[4])

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt

regp_methods_palette = {"Concentration-Noisy": "g", "Constant-Noisy": "b", "Spatial-Noisy": "k", "None-Noisy": "r"}
sequential_strategies_palette = {"UCB10": "solid"}

def get_key_value(regp_method, test_function, sequential_strategy):
    key = "{} ({})".format(regp_method, sequential_strategy)
    value = [
        os.path.join(data_dir, sequential_strategy, test_function, regp_method),
        (regp_methods_palette[regp_method], sequential_strategies_palette[sequential_strategy])
    ]

    return key, value

def plot_test_function(test_function, sequential_strategy, args_list, n_runs, log_scale, y_label):
    palette = {}
    for regp_method in ["Spatial-Noisy", "None-Noisy"]:
        key, value = get_key_value(regp_method, test_function, sequential_strategy)
        palette[key] = value

    plt.subplot(*args_list[0])
    plot_noisy_optim(
        palette,
        300,
        test_function,
        n_runs,
        10,
        y_label
    )
    plt.tight_layout()

    plt.subplot(*args_list[1])
    plot_value_of_estimated_minimizer(
        palette,
        300,
        test_function,
        n_runs,
        10,
        y_label,
        log_scale
    )
    plt.tight_layout()

    plt.subplot(*args_list[2])
    plot_error_on_estimated_minimizer(
        palette,
        300,
        test_function,
        n_runs,
        10,
        y_label,
        log_scale
    )
    plt.tight_layout()

test_functions = ["noisy-goldsteinprice-10000.0", "noisy-goldstein_price_log-9.0"]

plt.subplots(3, 2, sharex=True, figsize=(6, 6))
plot_test_function(test_functions[0], sequential_strategy, [[3, 2, 1], [3, 2, 3], [3, 2, 5]], n_runs, True, y_label=True)
plot_test_function(test_functions[1], sequential_strategy, [[3, 2, 2], [3, 2, 4], [3, 2, 6]], n_runs, True, y_label=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "multi_noisy.pgf"))