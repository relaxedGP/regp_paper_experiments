import sys, os
from plotting_utils import investigate_multi_modal_optim, get_test_function_format
import gpmpcontrib.optim.test_problems
from scipy.optimize import minimize
from scipy.stats import qmc
import numpy as np

sequential_strategy = sys.argv[1]
data_dir = sys.argv[2]
test_function = sys.argv[3]
if len(sys.argv) == 5:
    force_no_minima = sys.argv[4] == "True"
else:
    force_no_minima = False

import matplotlib.pyplot as plt

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}
sequential_strategies_palette = {"EI": "solid", "UCB10": "solid"}

# These lists contain values appearing repeatedly in the results. Only the cases for which there was a small number
# of highly recurrent values are considered.
local_minima_lists = {
    "hartman6": [-3.20316191],
    "goldstein_price_log": [4.4308168, 3.40119738],
    "shekel5": [-5.055197728765767, -2.6828603956634143],
    "shekel7": [-5.0876717219047345]
}
global_minimums = {
    "hartman6": -3.32236801,
    "goldstein_price_log": 1.09861229,
    "shekel5": -10.15319967864548,
    "shekel7": -10.402915336170734
}

def get_key_value(regp_method, test_function, sequential_strategy):
    key = "{} ({})".format(regp_method, sequential_strategy)
    value = [
        os.path.join(data_dir, sequential_strategy, test_function, regp_method),
        (regp_methods_palette[regp_method], sequential_strategies_palette[sequential_strategy])
    ]

    return key, value

def plot_test_function(test_function, sequential_strategy, local_minima_list, global_minimum):
    palette = {}
    for regp_method in ["None", "Constant", "Spatial", "Concentration"]:
        key, value = get_key_value(regp_method, test_function, sequential_strategy)
        palette[key] = value

    investigate_multi_modal_optim(
        palette,
        300,
        test_function,
        100,
        10,
        local_minima_list,
        global_minimum
    )

#
n_run_local_opt = 100

problem = getattr(gpmpcontrib.optim.test_problems, test_function)
box = problem.input_box
assert all([len(_v) == len(box[0]) for _v in box])
bounds = [tuple(box[i][k] for i in range(len(box))) for k in range(len(box[0]))]

funs = []

for _ in range(n_run_local_opt):
    x0_std = np.random.uniform(size=problem.input_dim).reshape(1, -1)
    x0 = qmc.scale(x0_std, box[0], box[1]).ravel()
    _fun = minimize(fun=lambda x: problem.eval(x.reshape(1, -1)), bounds=bounds, x0=x0).fun
    funs.append(_fun)

funs.sort()
print("Local opt output: ", funs)

#
if (test_function in local_minima_lists.keys() and test_function in global_minimums.keys()) and not force_no_minima:
    local_minima_list = local_minima_lists[test_function]
    global_minimum = global_minimums[test_function]
else:
    local_minima_list = None
    global_minimum = None

plt.figure()
plt.title(get_test_function_format(test_function))

plot_test_function(test_function, sequential_strategy, local_minima_list, global_minimum)

plt.tight_layout()
plt.show()
