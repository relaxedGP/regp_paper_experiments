import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

def get_test_function_format(x):

    x_split = x.split('_')

    assert len(x_split) <= 2

    if x_split[0] == 'GoldsteinPrice':
        func_name = 'Goldstein-Price'
    elif x_split[0] == 'GoldsteinPriceLog':
        func_name = 'Log-Goldstein-Price'
    elif x_split[0] == 'GoldsteinPriceGeomAvg':
        func_name = 'G-P'
    elif x_split[0] == 'GoldsteinPriceGeomAvgLog':
        func_name = 'G-P Log'
    elif x_split[0] == 'DixonPrice':
        func_name = 'Dixon-Price'
    elif x == 'CrossInTray':
        func_name = 'Cross-in-Tray'
    elif x == 'CamelBack':
        func_name = 'Six-hump Camel'
    elif x == 'ThreeHumpCamelBack':
        func_name = 'Three-hump Camel'
    elif 'Hartman' in x_split[0]:
        func_name = 'Hartman $({})$'.format(x_split[0][7])
    elif 'Shekel' in x_split[0]:
        func_name = x
    elif x_split[0] == 'Sin2':
        func_name = x
    else:
        func_name = x_split[0]

    if len(x_split) > 1:
        func_name = "{} $({})$".format(func_name, x_split[1])

    return func_name

def get_min_targets_dict():
    return {
         'GoldsteinPrice': 4.630657803991099,
         'GoldsteinPriceLog': 1.53269893,
         'Hartman3': -3.8579851379549845,
         'Hartman6': -3.0718173831021325,
         'Branin': 0.39947399092611224,
         'Shekel5': -3.432734952853385,
         'Shekel7': -3.029772054726411,
         'Shekel10': -3.688000593866475,
         'Sin2': -0.8507671748132886,
         'CamelBack': -1.0309776951593344,
         'Rosenbrock_4': 45.42394451416677,
         'Rosenbrock_6': 632.8917633859533,
         'Rosenbrock_10': 4918.500256153549,
         'Ackley_4': 18.44103073120918,
         'Ackley_6': 16.076844501272827,
         'Ackley_10': 15.374008736567806,
         'ThreeHumpCamelBack': 0.0006440883906866736,
         'CrossInTray': -1.934414910730356,
         'Beale': 0.09297149626811346,
         'DixonPrice_4': 162.0984877203831,
         'DixonPrice_6': 111.40289721671844,
         'DixonPrice_10': 850.3008792336834,
         'Perm_4': 199.42446005482032,
         'Perm_6': 1580986.9769695974,
         'Perm_10': 2.2145398423924444e+18,
         'Michalewicz_4': -3.1048964619461716,
         'Michalewicz_6': -3.1536227753080497,
         'Michalewicz_10': -3.700359579501645,
         'Zakharov_4': 30.044662757124485,
         'Zakharov_6': 118.91356256273377,
         'Zakharov_10': 17393.98579965569
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

    cummin = {k: fetch_data(palette[k][0], n_runs) for k in palette.keys()}

    for k in cummin.keys():
        cummin[k] = np.array([np.minimum.accumulate(np.concatenate((_l, np.array((max_f_evals - _l.shape[0]) * [np.inf])))) for _l in cummin[k]]).mean(0)

    plt.figure(figsize=(3.0, 2.6))

    plt.title(get_test_function_format(test_function))

    for k in cummin.keys():
        plt.plot(cummin[k], label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    plt.legend()

    plt.tight_layout()

    plt.show()