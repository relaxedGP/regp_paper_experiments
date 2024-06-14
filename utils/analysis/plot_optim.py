import sys, os
from plotting_utils import plot_cummin

sequential_strategy = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]
if len(sys.argv) == 5:
    test_function = sys.argv[4]
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

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}
sequential_strategies_palette = {"EI": "solid", "UCB10": "solid"}

def get_key_value(regp_method, test_function, sequential_strategy):
    key = "{} ({})".format(regp_method, sequential_strategy)
    value = [
        os.path.join(data_dir, sequential_strategy, test_function, regp_method),
        (regp_methods_palette[regp_method], sequential_strategies_palette[sequential_strategy])
    ]

    return key, value

def plot_test_function(test_function, sequential_strategy):
    palette = {}
    for regp_method in ["Concentration", "Constant", "Spatial", "None"]:
        key, value = get_key_value(regp_method, test_function, sequential_strategy)
        palette[key] = value

    plot_cummin(
        palette,
        300,
        test_function,
        100
    )

test_functions = [
    "branin",
    "threehumpcamelback",
    "camel_back",
    "hartman3",
    "hartman6",
    "ackley4",
    "ackley6",
    "ackley10",
    "rosenbrock4",
    "rosenbrock6",
    "rosenbrock10",
    "shekel5",
    "shekel7",
    "shekel10",
    "goldsteinprice",
    "goldstein_price_log",
    "crossintray",
    "beale",
    "dixon_price4",
    "dixon_price6",
    "dixon_price10",
    "perm4",
    "perm6",
    "perm10",
    "michalewicz4",
    "michalewicz6",
    "michalewicz10",
    "zakharov4",
    "zakharov6",
    "zakharov10"
]

if test_function is not None:
    test_functions = [test_function]
    show = True
else:
    show = False

for test_function in test_functions:
    plot_test_function(test_function, sequential_strategy)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "{}.pgf".format(test_function)))
        plt.close()