import sys, os
from plotting_utils import plotter_levelset

test_function = sys.argv[1]
data_dir = sys.argv[2]

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}
sequential_strategies_palette = {"straddle": "solid"}
sequential_strategy = "straddle"

def get_key_value(regp_method, test_function, sequential_strategy):
    key = "{} ({})".format(regp_method, sequential_strategy)
    value = [
        os.path.join(data_dir, sequential_strategy, test_function, regp_method),
        (regp_methods_palette[regp_method], sequential_strategies_palette[sequential_strategy])
    ]

    return key, value

palette = {}
for regp_method in ["Concentration", "Constant", "Spatial", "None"]:
    key, value = get_key_value(regp_method, test_function, sequential_strategy)
    palette[key] = value

plotter_levelset(
    palette,
    30
)

