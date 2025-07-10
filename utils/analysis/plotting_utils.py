import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import gpmpcontrib.optim.test_problems as optim_test_problems
import gpmpcontrib.levelset.test_problems as levelset_test_problems


def get_func_param(x):
    param = x[-1]
    if param == "0":
        param = x[-2:]
    return param

def get_test_function_format(x):

    if x == "g10c6mod":
        return x
    elif x == "g10c6modmod":
        return x
    elif x == "goldsteinprice-1000":
        return 'Goldstein-Price'
    elif x == "goldstein_price_log-6.90775":
        return 'Log-Goldstein-Price'

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
    elif x.split("-")[0] == "noisy_goldstein_price":
        noise_std = np.sqrt(float(x.split("-")[1]))
        if noise_std.is_integer():
            noise_std = round(noise_std)
        return r"Noisy Goldstein-Price ($\eta = {}$)".format(noise_std)
    elif x.split("-")[0] == "noisy_goldstein_price_log":
        noise_std = np.sqrt(float(x.split("-")[1]))
        if noise_std.is_integer():
            noise_std = round(noise_std)
        return r"Noisy Log-Goldstein-Price ($\eta = {}$)".format(noise_std)
    elif x.split("-")[0] == "noisy_beale":
        noise_std = np.sqrt(float(x.split("-")[1]))
        if noise_std.is_integer():
            noise_std = round(noise_std)
        return r"Noisy Beale ($\eta = {}$)".format(noise_std)
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
    if test_function == 'goldstein_price_log':
        x_array, targets = get_spatial_quantiles_targets('goldsteinprice')
        return x_array, np.log(targets).tolist()

    path = os.path.join("spatial_quantiles", test_function)

    targets = pd.read_csv(os.path.join(path, 'thresholds.csv'), header=None).values.ravel()
    x_array = pd.read_csv(os.path.join(path, 'proba.csv'), header=None).values.ravel()

    # minimum_target = get_min_targets_dict()[test_function]

    # key = targets <= minimum_target

    # targets = targets[key].tolist()
    # x_array = x_array[key].tolist()

    return x_array, targets

def fetch_data(data_dir, n_runs):
    L = []
    for i in range(n_runs):
        sub_path = os.path.join(data_dir, 'data_{}.npy'.format(i))

        data = np.load(sub_path)[:, -1]

        L.append(data)

    return L

def fetch_value_estimated_min_propeties(data_dir, n_runs):
    L_truth = []
    L_estimated = []
    for i in range(n_runs):
        sub_path_truth = os.path.join(data_dir, 'truevalue_{}.npy'.format(i))
        sub_path_estimated = os.path.join(data_dir, 'estimatedvalue_{}.npy'.format(i))

        data_truth = np.load(sub_path_truth).ravel()
        data_estimated = np.load(sub_path_estimated).ravel()

        L_truth.append(data_truth)
        L_estimated.append(data_estimated)

    return L_truth, L_estimated

def get_cummins(data_dir, n_runs, max_f_evals):
    runs_data = fetch_data(data_dir, n_runs)

    cummins = []

    for _l in runs_data:
        inf_padding = (max_f_evals - _l.shape[0]) * [np.inf]
        inf_padding = np.array(inf_padding)

        _padded_l = np.concatenate((_l, inf_padding))

        cummin = np.minimum.accumulate(_padded_l)

        cummins.append(cummin)

    cummins = np.array(cummins)

    return cummins

def get_cummin_statistics(data_dir, n_runs, max_f_evals):
    cummins = get_cummins(data_dir, n_runs, max_f_evals)
    return np.quantile(cummins, 0.1, axis=0), np.quantile(cummins, 0.5, axis=0), np.quantile(cummins, 0.9, axis=0)

def get_value_of_estimated_min_statistics(data_dir, n_runs, max_f_evals):
    truth_data, _ = fetch_value_estimated_min_propeties(data_dir, n_runs)
    truth_data = np.array(truth_data)
    return np.quantile(truth_data, 0.1, axis=0), np.quantile(truth_data, 0.5, axis=0), np.quantile(truth_data, 0.9, axis=0)

def get_error_estimated_min_statistics(data_dir, n_runs, max_f_evals):
    truth_data, estimated_data = fetch_value_estimated_min_propeties(data_dir, n_runs)
    truth_data = np.array(truth_data)
    estimated_data = np.array(estimated_data)
    diff_data = truth_data - estimated_data
    return np.quantile(np.abs(diff_data), 0.5, axis=0)

def get_optim_statistics(data_dir, n_runs, max_f_evals):
    values = fetch_data(data_dir, n_runs)

    assert all([_values.shape == values[0].shape for _values in values]), [_values.shape for _values in values]
    # for i in range(len(values)):
    #     _values = values[i]
    #     if len(_values) < max_f_evals:
    #         inf_padding = (max_f_evals - _values.shape[0]) * [np.inf]
    #         inf_padding = np.array(inf_padding)
    #
    #         _values = np.concatenate((_values, inf_padding))
    #         values[i] = _values

    values = np.array(values)

    # # Expose?
    # n_bootstrap = 100
    #
    # lower_q = np.zeros([values.shape[1]])
    # median = np.zeros([values.shape[1]])
    # upper_q = np.zeros([values.shape[1]])
    #
    # for i in range(values.shape[1]):
    #     _medians = []
    #     for j in range(n_bootstrap):
    #         samples = np.random.choice(values[:, i], size=values[:, i].shape[0], replace=True)
    #         _medians.append(np.quantile(samples, 0.5))
    #     lower_q[i] = np.quantile(_medians, 0.1)
    #     median[i] = np.quantile(_medians, 0.5)
    #     upper_q[i] = np.quantile(_medians, 0.9)

    return np.quantile(values, 0.1, axis=0), np.quantile(values, 0.5, axis=0), np.quantile(values, 0.9, axis=0)

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
        n0_over_dim
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    # Get dimension
    problem = getattr(optim_test_problems, test_function)
    dim = problem.input_dim

    # Get spatial quantiles
    x_array, targets = get_spatial_quantiles_targets(test_function)

    cummin = {k: get_cummin_statistics(palette[k][0], n_runs, max_f_evals) for k in palette.keys()}

    # level = 0.66

    best_perf = np.inf

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(3.0, 2.6))
    ax1, ax2 = ax

    plt.suptitle(get_test_function_format(test_function))

    # First plot
    interp = lambda x: np.interp(x, np.flip(targets), np.flip(x_array))

    for k in cummin.keys():
        lower_q, med, upper_q = cummin[k]

        # Filter out initial DoE
        lower_q = lower_q[n0_over_dim * dim:]
        med = med[n0_over_dim * dim:]
        upper_q = upper_q[n0_over_dim * dim:]

        #
        assert lower_q.min() > min(targets)

        # Fetch best perf
        best_perf = min(best_perf, lower_q.min())

        # Interpolate quantiles
        lower_q = interp(lower_q)
        med = interp(med)
        upper_q = interp(upper_q)

        # Plot
        abscissa = list(range(n0_over_dim * dim, max_f_evals))
        ax1.fill_between(abscissa, lower_q, upper_q, color=palette[k][1][0], alpha=0.2)
        ax1.plot(abscissa, med, label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    ax1.semilogy()

    key_filter = (targets >= best_perf)

    # Second plot
    props = {k: get_format_data(palette[k][0], targets, max_f_evals, n_runs)[0] for k in palette.keys()}
    for k in palette.keys():
        _props_array = np.array(props[k])
        ax2.plot(_props_array[key_filter], x_array[key_filter], label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    # ax2.axvline(level, color='k', linestyle='dashed')

    ax2.set_xlim(-0.1, 1.1)

    ax2.set_xticks(
        np.array([0, 0.33, 0.66, 1.0]),
        [
            matplotlib.text.Text(0.0, 0, '$\\mathdefault{0.0}$'),
            matplotlib.text.Text(0.33, 0, '$\\mathdefault{0.33}$'),
            matplotlib.text.Text(0.66, 0, '$\\mathdefault{0.66}$'),
            matplotlib.text.Text(1.0, 0, '$\\mathdefault{1.0}$'),
        ]
    )

    # ax2.invert_xaxis()
    # ax2.semilogx()

def plot_value_of_estimated_minimizer(
        palette,
        max_f_evals,
        test_function,
        n_runs,
        n0_over_dim
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    # Get dimension
    # FIXME:() Dirty
    if test_function.split("-")[0] == "noisy_goldstein_price":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_goldstein_price(noise_variance, None)
        global_minimum = 3.0
        noiseless_test_function_name = "goldsteinprice"
    elif test_function.split("-")[0] == "noisy_goldstein_price_log":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_goldstein_price_log(noise_variance, None)
        global_minimum = np.log(3.0)
        noiseless_test_function_name = "goldstein_price_log"
    elif test_function.split("-")[0] == "noisy_beale":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_beale(noise_variance, None)
        global_minimum = 0.0
        noiseless_test_function_name = "beale"
    else:
        problem = getattr(optim_test_problems, test_function)

    # Get spatial quantiles
    x_array, targets = get_spatial_quantiles_targets(noiseless_test_function_name)

    value_of_estimated_min = {k: get_value_of_estimated_min_statistics(palette[k][0], n_runs, max_f_evals)
                              for k in palette.keys()}

    dim = problem.input_dim

    # level = 0.66

    best_perf = np.inf

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(3.0, 2.6))
    ax1 = ax

    plt.suptitle(get_test_function_format(test_function))

    # First plot
    interp = lambda x: np.interp(x, np.flip(targets), np.flip(x_array))

    for k in value_of_estimated_min.keys():
        lower_q, med, upper_q = value_of_estimated_min[k]

        # # Filter out initial DoE
        # lower_q = lower_q[n0_over_dim * dim:]
        # med = med[n0_over_dim * dim:]
        # upper_q = upper_q[n0_over_dim * dim:]

        #
        assert lower_q.min() > min(targets)

        # Fetch best perf
        best_perf = min(best_perf, lower_q.min())

        # Interpolate quantiles
        lower_q = interp(lower_q)
        med = interp(med)
        upper_q = interp(upper_q)

        # Plot
        abscissa = list(range(n0_over_dim * dim, max_f_evals))
        ax1.fill_between(abscissa, lower_q, upper_q, color=palette[k][1][0], alpha=0.2)
        ax1.plot(abscissa, med, label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    ax1.semilogy()

    plt.axhline(interp(global_minimum + np.sqrt(noise_variance)), color="orange", linestyle="dashed")
    plt.axhline(interp(global_minimum + 2 * np.sqrt(noise_variance)), color="orange", linestyle="dashed")

def plot_error_on_estimated_minimizer(
        palette,
        max_f_evals,
        test_function,
        n_runs,
        n0_over_dim
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    # Get dimension
    # FIXME:() Dirty
    if test_function.split("-")[0] == "noisy_goldstein_price":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_goldstein_price(noise_variance, None)
        global_minimum = 3.0
        noiseless_test_function_name = "goldsteinprice"
    elif test_function.split("-")[0] == "noisy_goldstein_price_log":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_goldstein_price_log(noise_variance, None)
        global_minimum = np.log(3.0)
        noiseless_test_function_name = "goldstein_price_log"
    elif test_function.split("-")[0] == "noisy_beale":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_beale(noise_variance, None)
        global_minimum = 0
        noiseless_test_function_name = "beale"
    else:
        problem = getattr(optim_test_problems, test_function)

    # Get spatial quantiles
    # x_array, targets = get_spatial_quantiles_targets(noiseless_test_function_name)

    error_on_estimated_min = {k: get_error_estimated_min_statistics(palette[k][0], n_runs, max_f_evals)
                              for k in palette.keys()}

    dim = problem.input_dim

    # # level = 0.66
    #
    # best_perf = np.inf

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(3.0, 2.6))
    ax1 = ax

    plt.suptitle(get_test_function_format(test_function))

    ## First plot
    # interp = lambda x: np.interp(x, np.flip(targets), np.flip(x_array))

    for k in error_on_estimated_min.keys():
        abs_med = error_on_estimated_min[k]

        # # Filter out initial DoE
        # lower_q = lower_q[n0_over_dim * dim:]
        # med = med[n0_over_dim * dim:]
        # upper_q = upper_q[n0_over_dim * dim:]

        #
        # assert lower_q.min() > min(targets)

        # Fetch best perf
        # best_perf = min(best_perf, lower_q.min())

        # # Interpolate quantiles
        # lower_q = interp(lower_q)
        # med = interp(med)
        # upper_q = interp(upper_q)

        # Plot
        abscissa = list(range(n0_over_dim * dim, max_f_evals))
        # ax1.fill_between(abscissa, lower_q, upper_q, color=palette[k][1][0], alpha=0.2)
        ax1.plot(abscissa, abs_med, label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

        plt.semilogy()

def plot_noisy_optim(
        palette,
        max_f_evals,
        test_function,
        upper_threshold,
        n_runs,
        n0_over_dim
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    # Get dimension
    # FIXME:() Dirty
    if test_function.split("-")[0] == "noisy_goldstein_price":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_goldstein_price(noise_variance, None)
        global_minimum = 3.0
    elif test_function.split("-")[0] == "noisy_goldstein_price_log":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_goldstein_price_log(noise_variance, None)
        global_minimum = np.log(3.0)
    elif test_function.split("-")[0] == "noisy_beale":
        noise_variance = float(test_function.split("-")[1])
        problem = optim_test_problems.noisy_beale(noise_variance, None)
        global_minimum = 0.0
    else:
        problem = getattr(optim_test_problems, test_function)

    dim = problem.input_dim

    statistics = {k: get_optim_statistics(palette[k][0], n_runs, max_f_evals) for k in palette.keys()}

    plt.figure(figsize=(3.0, 2.6))
    plt.title(get_test_function_format(test_function))

    for k in statistics.keys():
        lower_q, med, upper_q = statistics[k]

        # Filter out initial DoE
        lower_q = lower_q[n0_over_dim * dim:]
        med = med[n0_over_dim * dim:]
        upper_q = upper_q[n0_over_dim * dim:]

        # Plot
        abscissa = list(range(n0_over_dim * dim, max_f_evals))
        plt.fill_between(abscissa, lower_q, upper_q, color=palette[k][1][0], alpha=0.2)
        plt.plot(abscissa, med, label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

        if upper_threshold is not None:
            plt.ylim(lower_q.min(), upper_threshold)

    plt.axhline(global_minimum + np.sqrt(noise_variance), color="orange", linestyle="dashed")
    plt.axhline(global_minimum + 2 * np.sqrt(noise_variance), color="orange", linestyle="dashed")

def investigate_multi_modal_optim(
        palette,
        max_f_evals,
        test_function,
        n_runs,
        n0_over_dim,
        local_minima_list=None,
        global_minimum=None
    ):

    # Get dimension
    problem = getattr(optim_test_problems, test_function)
    dim = problem.input_dim

    # Get spatial quantiles
    x_array, targets = get_spatial_quantiles_targets(test_function)

    cummins_full = {k: get_cummins(palette[k][0], n_runs, max_f_evals) for k in palette.keys()}
    cummins_stats = {k: get_cummin_statistics(palette[k][0], n_runs, max_f_evals) for k in palette.keys()}

    for k in cummins_full.keys():
        print("\n", k)
        lower_q, med, upper_q = cummins_stats[k]
        cummins = cummins_full[k]

        if local_minima_list is None or global_minimum is None:
            print("Best values found: {}".format(cummins[:, max_f_evals - 1].tolist()))

        # Filter out initial DoE
        lower_q = lower_q[n0_over_dim * dim:]
        med = med[n0_over_dim * dim:]
        upper_q = upper_q[n0_over_dim * dim:]
        cummins = cummins[:, n0_over_dim * dim:]

        #
        assert lower_q.min() > min(targets)

        # Plot
        abscissa = list(range(n0_over_dim * dim, max_f_evals))
        plt.fill_between(abscissa, lower_q, upper_q, color=palette[k][1][0], alpha=0.2)
        plt.plot(abscissa, med, label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

        for i in range(n_runs):
            plt.plot(abscissa, cummins[i, :], color=palette[k][1][0], linestyle="dashed")

        if local_minima_list is not None and global_minimum is not None:
            global_minimum_reached_cpt = 0
            for i in range(n_runs):
                final_val = cummins[i, :].min()
                if all([np.abs(final_val - _l) > np.abs(final_val - global_minimum) for _l in local_minima_list]):
                    global_minimum_reached_cpt += 1

            print("Frac global minimum reached: {}".format(global_minimum_reached_cpt/n_runs))

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
        values = []
        for d in L:
            if i <= d.shape[0] - 1:
                values.append(d[i])
            else:
                print("Sym diff vol array: {} has less than {} elements.".format(d, max_idx))

        res.append([np.quantile(values, 0.1), np.quantile(values, 0.5), np.quantile(values, 0.9)])

    return np.array(res)



def plotter_levelset(
        palette,
        n_runs,
        test_function,
        max_f_evals,
        n0_over_dim
    ):
    """
    palette is a dict like: {"Concentration": [path, ("green", "solid")], ...}
    """

    plt.title(get_test_function_format(test_function))

    # Get dimension
    problem = getattr(levelset_test_problems, test_function)
    dim = problem.input_dim

    # Fetch data
    data = {k: aggregate_levelset_data(palette[k][0], n_runs) for k in palette.keys()}

    abscissa = list(range(n0_over_dim * dim, max_f_evals))
    for k in palette.keys():
        res_k = data[k]
        plt.fill_between(abscissa, res_k[:, 0], res_k[:, 2], color=palette[k][1][0], alpha=0.2)
        plt.plot(abscissa, res_k[:, 1], label=k, linestyle=palette[k][1][1], color=palette[k][1][0])

    plt.ylim([0.0004, 1])
    plt.semilogy()

    # plt.legend()

    plt.tight_layout()
