import sys, os
from plotting_utils import plot_noisy_optim, plot_value_of_estimated_minimizer, plot_error_on_estimated_minimizer

sequential_strategy = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]
if len(sys.argv) == 5:
    test_function = sys.argv[4]
    upper_threshold = None
elif len(sys.argv) == 6:
    test_function = sys.argv[4]
    upper_threshold = float(sys.argv[5])
else:
    test_function = None
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

def plot_test_function(test_function, upper_threshold, sequential_strategy, show):
    palette = {}
    for regp_method in ["Constant-Noisy", "Spatial-Noisy", "None-Noisy", "Concentration-Noisy"]:
        key, value = get_key_value(regp_method, test_function, sequential_strategy)
        palette[key] = value

    plot_noisy_optim(
        palette,
        300,
        test_function,
        upper_threshold,
        30,
        10
    )
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "raw-{}.pdf".format(test_function)))
        plt.close()

    plot_value_of_estimated_minimizer(
        palette,
        300,
        test_function,
        30,
        10
    )
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "perf_min-{}.pdf".format(test_function)))
        plt.close()

    plot_error_on_estimated_minimizer(
        palette,
        300,
        test_function,
        30,
        10
    )
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "error_min-{}.pdf".format(test_function)))
        plt.close()


test_functions = []

if test_function is not None:
    test_functions = [(test_function, upper_threshold)]
    show = True
else:
    show = False

for test_function in test_functions:
    plot_test_function(test_function[0], test_function[1], sequential_strategy, show)
