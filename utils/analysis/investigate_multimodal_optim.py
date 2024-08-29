import sys, os
from plotting_utils import investigate_multi_modal_optim, get_test_function_format

sequential_strategy = sys.argv[1]
data_dir = sys.argv[2]
test_function = sys.argv[3]

import matplotlib.pyplot as plt

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}
sequential_strategies_palette = {"EI": "solid", "UCB10": "solid"}

local_minima_lists = {"hartman6": [-3.20316191], "goldstein_price_log": [4.4308168, 3.40119738]}
global_minimums = {"hartman6": -3.32236801, "goldstein_price_log": 1.09861229}

def get_key_value(regp_method, test_function, sequential_strategy):
    key = "{} ({})".format(regp_method, sequential_strategy)
    value = [
        os.path.join(data_dir, sequential_strategy, test_function, regp_method),
        (regp_methods_palette[regp_method], sequential_strategies_palette[sequential_strategy])
    ]

    return key, value

def plot_test_function(test_function, sequential_strategy, local_minima_list, global_minimum):
    palette = {}
    for regp_method in ["Constant", "Spatial", "None", "Concentration"]:
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


if test_function in local_minima_lists.keys() and test_function in global_minimums.keys():
    local_minima_list = local_minima_lists[test_function]
    global_minimum = global_minimums[test_function]
else:
    local_minima_list = None
    global_minimum = None

if local_minima_list is not None and global_minimum is not None:
    plt.figure()
    plt.title(get_test_function_format(test_function))

plot_test_function(test_function, sequential_strategy, local_minima_list, global_minimum)

if local_minima_list is not None and global_minimum is not None:
    plt.tight_layout()
    plt.show()
