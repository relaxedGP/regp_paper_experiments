import numpy as np
import sys, os
import matplotlib.pyplot as plt

input_folder = sys.argv[1]

seq_strategy = sys.argv[2]

n_runs = int(sys.argv[3])

regp_methods_palette = {"Concentration": "g", "Constant": "b", "Spatial": "k", "None": "r"}

regp_methods_times = {}
regp_methods_relative_times = {}

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

for test_function in test_functions:
    for k in regp_methods_palette.keys():
        for i in range(n_runs):
            # if not os.path.exists(os.path.join(input_folder, seq_strategy, test_function, k, "success_{}".format(i))):
                # print(test_function, k, i)
            data = np.load(os.path.join(input_folder, seq_strategy, test_function, k, "data_{}.npy".format(i)))
            if data.shape[0] != 300:
                print(test_function, k, i, data.shape[0])
