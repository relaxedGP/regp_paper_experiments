import os

n_runs = 100

# Optim
optim_test_functions = [
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

optim_test_functions.reverse()

optim_seq_strats = ["EI", "UCB10"]

for optim_test_function in optim_test_functions:
    for optim_seq_strat in optim_seq_strats:
        os.system(
            "bash run_allmethods.sh {} {} {}".format(optim_test_function, n_runs, optim_seq_strat)
        )

# Levelset
levelset_seq_strat = "straddle"

levelset_test_functions = ["g10c6mod", "g10c6modmod", "goldsteinprice-1000", "goldstein_price_log-6.90775"]

for levelset_test_function in levelset_test_functions:
    os.system(
        "bash run_allmethods.sh {} {} {}".format(levelset_test_function, n_runs, levelset_seq_strat)
    )

