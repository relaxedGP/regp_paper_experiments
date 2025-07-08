import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import gpmp as gp
import gpmpcontrib as gpc
import sys
import os
import gpmpcontrib.optim.test_problems as test_problems
import gpmpcontrib.smc
import traceback
from numpy.random import SeedSequence, default_rng
from does.designs import maximinlhs, sobol
import scipy.stats
import gpmp.num as gnp
import gpmpcontrib.samplingcriteria as sampcrit

assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

# Hard coded, for safety.
n_runs_max = 1000

# Settings default values and types for different options
env_options = {
    "ALGO": ("EI", str),
    "OUTPUT_DIR": ("results", str),
    "N_EVALUATIONS": (300, int),
    "SLURM_ARRAY_TASK_ID": (None, int),
    "N_RUNS": (1, int),
    "PROBLEM": ("goldsteinprice", str),
    "N0_OVER_D": (10, int),
    "STRATEGY": ("None", str),
    "Q_STRATEGY": (0.25, float),
    "CRIT_OPT_METHOD": ("SLSQP", str),
    "RELAXED_INIT": ("quad_prog", str),
    "FTOL": (1e-06, float),
    "GTOL": (1e-05, float),
    "EPS": (1e-08, float),
    "MAXFUN": (15000, int),
    "MAXITER": (15000, int),
    "N_SMC": (1000, int),
    "MH_STEPS": (5, int),
    "SMC_METHOD": ("step_with_possible_restart", str),
}

# Rngs Definition
def get_rng(i, n_runs_max, entropy=42):
    assert i <= n_runs_max - 1, (n_runs_max, i)

    ss = SeedSequence(entropy)

    child_seeds = ss.spawn(n_runs_max)
    streams = [default_rng(s) for s in child_seeds]

    return streams[i]

# Initialization Function
def initialize_optimization(env_options):
    options = {}
    crit_optim_options = {}
    algo_options = {}
    smc_options = {"mh_params": {}}
    # Loop through the environment options
    for key, (default, value_type) in env_options.items():
        value = os.getenv(key, default)
        if value is not None:
            if key == "CRIT_OPT_METHOD":
                # Add to crit_optim_options
                crit_optim_options["method"] = value_type(value)
            elif key in [
                "RELAXED_INIT",
                "FTOL",
                "GTOL",
                "EPS",
                "MAXFUN",
                "MAXITER",
            ]:
                # Add to crit_optim_options
                crit_optim_options[key.lower()] = value_type(value)
            elif key == "SLURM_ARRAY_TASK_ID" and value is not None:
                idx_run_list = [value_type(value)]
            elif key == "N_RUNS" and "SLURM_ARRAY_TASK_ID" not in os.environ:
                idx_run_list = list(range(value_type(value)))
            elif key == "STRATEGY":
                options["threshold_strategy"] = value

                # Check if Q_STRATEGY is set, use its value if available, otherwise use default
                options["q_strategy_value"] = float(
                    os.getenv("Q_STRATEGY", env_options["Q_STRATEGY"][0])
                )

            elif key == "Q_STRATEGY":
                continue  # Handled with STRATEGY
            elif key in [
                "MH_STEPS",
            ]:
                smc_options["mh_params"][key.lower()] = value_type(value)
            elif key in [
                "N_SMC",
            ]:
                smc_options["n"] = value_type(value)
            elif key in [
                "SMC_METHOD",
            ]:
                algo_options[key.lower()] = value_type(value)
            elif key == "ALGO":
                options["algo"] = value
                if options["algo"] in ["straddle"]:
                    options["task"] = "levelset"
                else:
                    options["task"] = "optim"
            else:
                # Add to options directly
                options[key.lower()] = value_type(value)

    # Set crit_optim_options in options
    if crit_optim_options:
        if crit_optim_options["method"] == "SLSQP":
            crit_optim_keys = ["method", "relaxed_init", "ftol", "eps", "maxiter"]
        elif crit_optim_options["method"] == "L-BFGS-B":
            crit_optim_keys = ["method", "relaxed_init", "ftol", "gtol", "eps", "maxfun", "maxiter"]
        else:
            raise ValueError(crit_optim_options["method"])
        crit_optim_options = {k: crit_optim_options[k] for k in crit_optim_keys}
        options["crit_optim_options"] = crit_optim_options
    if smc_options:
        algo_options["smc_options"] = smc_options
    if algo_options:
        options["algo_options"] = algo_options

    if options["task"] == "optim":
        import gpmpcontrib.optim.test_problems as test_problems
    elif options["task"] == "levelset":
        import gpmpcontrib.levelset.test_problems as test_problems
    else:
        raise ValueError(options["task"])

    # FIXME:() Dirty
    if "noisy_" in options["problem"]:
        noise_variance = float(options["problem"].split("-")[1])
        problem_name = options["problem"].split("-")[0]
        problem = getattr(test_problems, problem_name)(noise_variance)
    else:
        problem = getattr(test_problems, options["problem"])

    return problem, options, idx_run_list

def get_algo(problem, model, options):
    algo_name = options["algo"]
    if algo_name == "EI":
        import gpmpcontrib.optim.expectedimprovement as ei
        algo = ei.ExpectedImprovement(problem, model, options=options["algo_options"])
    elif algo_name[:3] == "UCB":
        q_UCB = 0.01 * float(algo_name[3:])
        print("UCB quantile: {}".format(q_UCB))
        import gpmpcontrib.optim.ucb as ucb
        algo = ucb.UCB(q_UCB, problem, model, options=options["algo_options"])
    elif algo_name ==  "straddle":
        t = get_levelset_threshold(problem)
        import gpmpcontrib.levelset.straddle as straddle
        algo = straddle.Straddle(t, problem, model, options=options["algo_options"])
    else:
        raise ValueError(algo_name)

    algo.force_param_initial_guess = True
    return algo

def get_levelset_threshold(problem):
    functions = problem.functions

    assert len(functions) == 1, functions

    bounds = functions[0]["bounds"]

    assert len(bounds) == 1, bounds
    assert len(bounds[0]) == 2, bounds
    assert bounds[0][0] == -np.inf, bounds

    return bounds[0][1]

# --------------------------------------------------------------------------------------
problem, options, idx_run_list = initialize_optimization(env_options)


# Repetition Loop
for i in idx_run_list:
    rng = get_rng(i, n_runs_max)
    options["algo_options"]["smc_options"]["rng"] = rng

    ni0 = options["n0_over_d"] * problem.input_dim
    xi = maximinlhs(problem.input_dim, ni0, problem.input_box, rng)

    if "None" in options["threshold_strategy"]:
        if options["threshold_strategy"] == "None":
            model_class = gpc.Model_ConstantMeanMaternpML
        elif options["threshold_strategy"] == "None-Noisy":
            model_class = gpc.NoisyModel_ConstantMeanMaternpML
        else:
            raise ValueError(options["threshold_strategy"])

        model = model_class(
            "GP_bench_{}".format(options["problem"]),
            output_dim=problem.output_dim,
            covariance_params={"p": 2},
            rng=rng,
            box=problem.input_box
        )
    else:
        threshold_strategy_splitted = options["threshold_strategy"].split("-")
        threshold_strategy_params = {
            "strategy": threshold_strategy_splitted[0],
            "level": options["q_strategy_value"],
            "n_init": ni0,
            "task": options["task"]
        }
        if options["task"] == "levelset":
            threshold_strategy_params["t"] = get_levelset_threshold(problem)

        if len(threshold_strategy_splitted) == 1:
            model_class = gpc.Model_ConstantMeanMaternp_reGP
        else:
            assert len(threshold_strategy_splitted) == 2 and threshold_strategy_splitted[1] == "Noisy", threshold_strategy_splitted
            model_class = gpc.NoisyModel_ConstantMeanMaternp_reGP

        model = model_class(
            threshold_strategy_params,
            "reGP_bench_{}".format(options["problem"]),
            crit_optim_options=options["crit_optim_options"],
            output_dim=problem.output_dim,
            covariance_params={"p": 2},
            rng=rng,
            box=problem.input_box
        )

    times_records = []

    algo = get_algo(problem, model, options)

    algo.set_initial_design(xi=xi)
    times_records.append(algo.training_time)

    n_iterations = options["n_evaluations"] - ni0
    assert n_iterations > 0

    meanparam_list = []
    covparam_list = []

    # TODO:() Do better.
    if "noisy_" in options["problem"]:
        true_value_list = []
        estimated_value_list = []

    if options["task"] == "levelset":
        m = 17
        sobol_sequence = sobol(problem.input_dim, m, problem.input_box)
        sym_diff_vol = []

    # Optimization loop
    for step_ind in range(n_iterations):
        print(f"\niter {step_ind}")

        # Run a step of the algorithm
        try:
            algo.step()
            times_records.append(algo.training_time)

            if options["task"] == "optim":
                print(
                    "Evaluation: {}, previous minimum: {}".format(
                        algo.zi.numpy().ravel()[-1],
                        algo.zi.numpy().ravel()[:-1].min()
                    )
                )

            if options["task"] == "levelset":
                mu_hat, var_hat = algo.predict(sobol_sequence)
                mu_hat = mu_hat.ravel()
                var_hat = var_hat.ravel()

                truth = problem.eval(sobol_sequence).ravel()

                t = algo.t

                gaussian_cdf = scipy.stats.norm.cdf(
                    (t - mu_hat)/np.sqrt(var_hat)
                )

                key_truth = (truth <= t).astype(int)

                exp_sym_diff = key_truth * (1 - gaussian_cdf) + (1 - key_truth) * gaussian_cdf
                sym_diff_vol.append(exp_sym_diff.mean())

            # TODO:() Do better.
            if "noisy_" in options["problem"]:
                # TODO:() Do better.
                if options["problem"].split("-")[0] == "noisy_goldstein_price":
                    noiseless_problem_name = "goldsteinprice"
                elif options["problem"].split("-")[0] == "noisy_goldstein_price_log":
                    noiseless_problem_name = "goldstein_price_log"
                else:
                    raise ValueError(options["problem"])

                noiseless_problem = getattr(test_problems, noiseless_problem_name)

                smc = gpmpcontrib.smc.SMC(problem.input_box)

                def boxify_criterion(x):
                   input_box = gnp.asarray(problem.input_box)
                   b = sampcrit.isinbox(input_box, x)

                   res, _ = algo.predict(x, convert_out=False)

                   res = - res.flatten()

                   res = gnp.where(gnp.asarray(b), res, - gnp.inf)

                   return res

                smc.subset(
                   func=boxify_criterion,
                   target=np.inf,
                   p0=0.2,
                   xi=algo.xi,
                   debug=False
                )

                zpm, _ = algo.predict(smc.particles.x, convert_out=False)

                assert not gnp.isnan(zpm).any()

                x_new = smc.particles.x[gnp.argmin(gnp.asarray(zpm))].reshape(1, -1)

                # print(x_new)
                # print(algo.predict(x_new)[0])
                # print(problem.eval(x_new))

                #
                init = gnp.to_np(x_new).ravel()

                def crit_(x):
                   x_row = x.reshape(1, -1)
                   zpm, _ = algo.predict(x_row, convert_out=False)
                   return zpm[0, 0]

                crit_jit = gnp.jax.jit(crit_)

                dcrit = gnp.jax.jit(gnp.grad(crit_jit))

                box = problem.input_box
                assert all([len(_v) == len(box[0]) for _v in box])

                bounds = [tuple(box[_i][k] for _i in range(len(box))) for k in range(len(box[0]))]
                model_argmin = gp.kernel.autoselect_parameters(
                   init, crit_jit, dcrit, bounds=bounds
                )

                if gnp.numpy.isnan(model_argmin).any():
                   minimizer = init

                for _i in range(model_argmin.shape[0]):
                   if model_argmin[_i] < bounds[_i][0]:
                       model_argmin[_i] = bounds[_i][0]
                   if bounds[_i][1] < model_argmin[_i]:
                       model_argmin[_i] = bounds[_i][1]

                if crit_(model_argmin) < crit_(init):
                   minimizer = model_argmin
                else:
                   minimizer = init

                minimizer = gnp.asarray(minimizer.reshape(1, -1))

                # print(minimizer)
                # print(algo.predict(minimizer)[0])
                # print(problem.eval(minimizer))

                true_value_list.append(noiseless_problem.eval(minimizer))
                estimated_value_list.append(algo.predict(minimizer)[0][0, 0])

        except gp.num.GnpLinalgError as e:
            i_error_path = os.path.join(options["output_dir"], str(i))
            os.mkdir(i_error_path)
            np.savetxt(os.path.join(i_error_path, "A.csv"), e.env_dict["A"].numpy(), delimiter=",")
            np.savetxt(os.path.join(i_error_path, "xi.csv"), e.env_dict["xi"].numpy(), delimiter=",")
            np.savetxt(os.path.join(i_error_path, "covparam.csv"), e.env_dict["covparam"].numpy(), delimiter=",")
            raise e
        except gp.kernel.NonInvertibleInitCovMat as e:
            print("Aborting: {}".format(e))
            print(traceback.format_exc())
            break
        except gpmpcontrib.smc.StoppingError as e:
            print("Aborting: {}".format(e))
            print(traceback.format_exc())
            break

        # Log models
        covparam_list.append(algo.models[0]["model"].covparam.numpy())
        meanparam_list.append(algo.models[0]["model"].meanparam.numpy())

        # # Prepare output directory
        # logs_path = os.path.join(options["output_dir"], "logs_{}".format(str(i)))
        # if not os.path.exists(logs_path):
        #     os.mkdir(logs_path)
        #
        # if "R" in algo.models[0].keys():
        #     np.save(
        #         os.path.join(logs_path, "R_{}.npy".format(step_ind)),
        #         np.array(algo.models[0]["R"])
        #     )
        #     np.save(
        #         os.path.join(logs_path, "zi_relaxed_{}.npy".format(step_ind)),
        #         algo.model.zi_relaxed.ravel().numpy()
        #     )

        # Prepare output directory
        i_output_path = os.path.join(options["output_dir"], "data_{}.npy".format(str(i)))
        i_times_path = os.path.join(options["output_dir"], "times_{}.npy".format(str(i)))
        i_covparam_path = os.path.join(options["output_dir"], "covparam_{}.npy".format(str(i)))
        i_meanparam_path = os.path.join(options["output_dir"], "meanparam_{}.npy".format(str(i)))

        if os.path.exists(i_output_path):
            os.remove(i_output_path)
        if os.path.exists(i_times_path):
            os.remove(i_times_path)

        # Save data
        np.save(i_output_path, np.hstack((algo.xi, algo.zi)))
        np.save(i_times_path, np.array(times_records))

        # Save parameters
        np.save(i_meanparam_path, np.array(meanparam_list))
        np.save(i_covparam_path, np.array(covparam_list))

        if options["task"] == "levelset":
            sym_diff_vol_path = os.path.join(options["output_dir"], "sym_diff_vol_{}.npy".format(str(i)))
            if os.path.exists(sym_diff_vol_path):
                os.remove(sym_diff_vol_path)
            np.save(sym_diff_vol_path, np.array(sym_diff_vol))

        # TODO:() Do better.
        if "Noisy" in options["threshold_strategy"]:
            true_value_path = os.path.join(options["output_dir"], "truevalue_{}.npy".format(str(i)))
            if os.path.exists(true_value_path):
                os.remove(true_value_path)
            np.save(true_value_path, np.array(true_value_list))

            estimated_value_path = os.path.join(options["output_dir"], "estimatedvalue_{}.npy".format(str(i)))
            if os.path.exists(estimated_value_path):
                os.remove(estimated_value_path)
            np.save(estimated_value_path, np.array(estimated_value_list))

    # endfor

    # Write success file
    success_file_path = os.path.join(options["output_dir"], "success_{}".format(str(i)))
    if os.path.exists(success_file_path):
        raise RuntimeError("Output file {} already exists.".format(success_file_path))
    open(success_file_path, 'w').close()
