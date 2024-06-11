import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

def get_func_param(x):
    param = x[-1]
    if param == "0":
        param = x[-2:]
    return param

def get_test_function_format(x):

    if x == 'goldsteinprice':
        return 'Goldstein-Price'
    elif x == 'goldstein_price_log':
        return 'Log-Goldstein-Price'
    elif x == 'crossintray':
        return 'Cross-in-Tray'
    elif x == 'camel_back':
        return 'Six-hump Camel'
    elif x == 'threehumpcamelback':
        return 'Three-hump Camel'
    elif x == 'beale':
        return 'Beale'
    elif x == 'branin':
        return 'Branin'
    elif 'hartman' in x:
        func_name = "Hartman"
    elif 'shekel' in x:
        func_name = "Shekel"
    elif 'zakharov' in x:
        func_name = "Zakharov"
    elif 'michalewicz' in x:
        func_name = "Michalewicz"
    elif 'perm' in x:
        func_name = "Perm"
    elif 'dixon_price' in x:
        func_name = 'Dixon-Price'
    elif 'rosenbrock' in x:
        func_name = 'Rosenbrock'
    elif 'ackley' in x:
        func_name = 'Ackley'
    else:
        raise ValueError(x)

    res = '{} $({})$'.format(func_name, get_func_param(x))
    return res

def get_min_targets_dict():
    return {
         'goldsteinprice': 4.630657803991099,
         'goldstein_price_log': 1.53269893,
         'hartman3': -3.8579851379549845,
         'hartman6': -3.0718173831021325,
         'branin': 0.39947399092611224,
         'shekel5': -3.432734952853385,
         'shekel7': -3.029772054726411,
         'shekel10': -3.688000593866475,
         'camel_back': -1.0309776951593344,
         'rosenbrock4': 45.42394451416677,
         'rosenbrock6': 632.8917633859533,
         'rosenbrock10': 4918.500256153549,
         'ackley4': 18.44103073120918,
         'ackley6': 16.076844501272827,
         'ackley10': 15.374008736567806,
         'threehumpcamelback': 0.0006440883906866736,
         'crossintray': -1.934414910730356,
         'beale': 0.09297149626811346,
         'dixon_price4': 162.0984877203831,
         'dixon_price6': 111.40289721671844,
         'dixon_price10': 850.3008792336834,
         'perm4': 199.42446005482032,
         'perm6': 1580986.9769695974,
         'perm10': 2.2145398423924444e+18,
         'michalewicz4': -3.1048964619461716,
         'michalewicz6': -3.1536227753080497,
         'michalewicz10': -3.700359579501645,
         'zakharov4': 30.044662757124485,
         'zakharov6': 118.91356256273377,
         'zakharov10': 17393.98579965569
    }

def get_spatial_quantiles_targets(test_function):
    if test_function == 'GoldsteinPriceLog':
        x_array, targets = get_spatial_quantiles_targets('GoldsteinPrice')
        return x_array, np.log(targets).tolist()

    path = os.path.join("spatial_quantiles", test_function)

    targets = pd.read_csv(os.path.join(path, 'thresholds.csv'), header=None).values.ravel()
    x_array = pd.read_csv(os.path.join(path, 'proba.csv'), header=None).values.ravel()

    minimum_target = get_min_targets_dict()[test_function]

    key = targets <= minimum_target

    targets = targets[key].tolist()
    x_array = x_array[key].tolist()

    return x_array, targets

def fetch_data(data_dir, n_runs):
    L = []
    for i in range(n_runs):
        sub_path = os.path.join(data_dir, 'data_{}.npy'.format(i))

        data = np.load(sub_path)[:, -1]

        L.append(data)

    return L

def get_cummin_averages(data_dir, n_runs, max_f_evals):
    runs_data = fetch_data(data_dir, n_runs)

    cummins = []

    for _l in runs_data:
        inf_padding = (max_f_evals - _l.shape[0]) * [np.inf]
        inf_padding = np.array(inf_padding)

        _padded_l = np.concatenate((_l, inf_padding))

        cummin = np.minimum.accumulate(_padded_l)

        cummins.append(cummin)

    cummins = np.array(cummins)

    average_cummins = cummins.mean(0)

    return average_cummins

def get_format_data(data_dir, targets, max_f_evals, n_runs):
    runs_data = fetch_data(data_dir, n_runs)

    props_reached = []
    averaged_reached_means = []

    for i in range(len(targets)):
        target = targets[i]

        #
        key_reached = np.array([run_datum.min() <= target for run_datum in runs_data])

        props_reached.append(key_reached.mean())

        #
        averaged_reached = []

        for run_datum in runs_data:
            run_datum_cummin = np.minimum.accumulate(run_datum)

            if run_datum_cummin.min() <= target:
                averaged_reached_datum = (run_datum_cummin <= target).argmax()
            else:
                averaged_reached_datum = max_f_evals

            averaged_reached.append(averaged_reached_datum)

        averaged_reached_means.append(np.mean(averaged_reached))

    return props_reached, averaged_reached_means

def plotter(
        palette,
        max_f_evals,
        test_function,
        n_runs,
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    x_array, targets = get_spatial_quantiles_targets(test_function)

    averages = {k: get_format_data(palette[k][0], targets, max_f_evals, n_runs)[1] for k in palette.keys()}
    props = {k: get_format_data(palette[k][0], targets, max_f_evals, n_runs)[0] for k in palette.keys()}

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.0, 2.6))
    ax1, ax2 = ax

    plt.suptitle(get_test_function_format(test_function))

    level = 0.66

    for k in averages.keys():
        to_print = np.array(averages[k])

        key = np.array(props[k]) < level

        to_print[key] = np.inf

        ax1.plot(x_array, to_print, label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    # ax1.set_ylim(0, max_f_evals)

    for k in averages.keys():
        ax2.plot(x_array, props[k], label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    ax2.axhline(level, color='k', linestyle='dashed')

    ax2.set_ylim(-0.1, 1.1)

    ax2.set_yticks(
        np.array([0, 0.33, 0.66, 1.0]),
        [
            matplotlib.text.Text(0.0, 0, '$\\mathdefault{0.0}$'),
            matplotlib.text.Text(0.33, 0, '$\\mathdefault{0.33}$'),
            matplotlib.text.Text(0.66, 0, '$\\mathdefault{0.66}$'),
            matplotlib.text.Text(1.0, 0, '$\\mathdefault{1.0}$'),
        ]
    )

    ax2.invert_xaxis()
    ax2.semilogx()

    plt.tight_layout()

    plt.show()

def plot_cummin(
        palette,
        max_f_evals,
        test_function,
        n_runs,
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    cummin = {k: get_cummin_averages(palette[k][0], n_runs, max_f_evals) for k in palette.keys()}

    plt.figure(figsize=(3.0, 2.6))

    plt.title(get_test_function_format(test_function))

    for k in cummin.keys():
        plt.plot(cummin[k], label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    plt.legend()

    plt.tight_layout()

    plt.show()

def fetch_levelset_data(data_dir, n_runs):
    L = []
    for i in range(n_runs):
        sub_path = os.path.join(data_dir, 'sym_diff_vol_{}.npy'.format(i))

        data = np.load(sub_path)

        L.append(data)

    return L

def aggregate_levelset_data(data_dir, n_runs):
    L = fetch_levelset_data(data_dir, n_runs)

    res = []

    max_idx = max([d.shape[0] for d in L])

    for i in range(max_idx):
        sum_value = 0
        cpt = 0
        for d in L:
            if i <= d.shape[0] - 1:
                sum_value += d[i]
                cpt += 1
            else:
                print("Sym diff vol array: {} has less than {} elements.".format(d, max_idx))

        res.append(sum_value/cpt)

    return res

def plotter_levelset(
        palette,
        n_runs,
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    data = {k: aggregate_levelset_data(palette[k][0], n_runs) for k in palette.keys()}

    plt.figure(figsize=(3.0, 2.6))

    for k in palette.keys():
        plt.plot(data[k], label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    plt.legend()

    plt.tight_layout()

    plt.show()