import sys, os
from plotting_utils import plotter_levelset
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt

test_function = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]

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
for regp_method in ["Constant", "Spatial", "None", "Concentration"]:
    key, value = get_key_value(regp_method, test_function, sequential_strategy)
    palette[key] = value

plt.figure(figsize=(3.0, 2.6))

plotter_levelset(
    palette,
    100,
    test_function,
    300,
    10
)

plt.savefig(os.path.join(output_dir, "{}.pgf".format(test_function)))

# plt.show()
