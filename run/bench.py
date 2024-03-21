import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import gpmp as gp
import gpmpcontrib as gpc
import sys
import os
import gpmpcontrib.optim.test_problems as test_problems
import traceback
from numpy.random import SeedSequence, default_rng
from utils.does.designs import maximinlhs

assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

# Hard coded, for safety.
n_runs_max = 1000

# Settings default values and types for different options
env_options = {
    "ALGO": ("EI", str),
    "OUTPUT_DIR": ("output", str),
    "N_EVALUATIONS": (300, int),
    "SLURM_ARRAY_TASK_ID": (None, int),
    "N_RUNS": (1, int),
    "PROBLEM": ("goldsteinprice", str),
    "N0_OVER_D": (10, int),
    "STRATEGY": ("None", str),
    "Q_STRATEGY": (0.25, float),
    "CRIT_OPT_METHOD": ("SLSQP", str),
    "RELAXED_INIT": ("flat", str),
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
                    options["t"] = float(os.getenv("T"))
                else:
                    options["task"] = "optim"
            else:
                # Add to options directly
                options[key.lower()] = value_type(value)

    # Set crit_optim_options in options
    if crit_optim_options:
        options["crit_optim_options"] = crit_optim_options
    if smc_options:
        algo_options["smc_options"] = smc_options
    if algo_options:
        options["algo_options"] = algo_options

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
        t = options["t"]
        import gpmpcontrib.levelset.straddle as straddle
        algo = straddle.Straddle(t, problem, model, options=options["algo_options"])
    else:
        raise ValueError(algo_name)

    algo.force_param_initial_guess = True
    return algo

# --------------------------------------------------------------------------------------
problem, options, idx_run_list = initialize_optimization(env_options)


# Repetition Loop
for i in idx_run_list:
    rng = get_rng(i, n_runs_max)
    options["algo_options"]["smc_options"]["rng"] = rng

    ni0 = options["n0_over_d"] * problem.input_dim
    xi = maximinlhs(problem.input_dim, ni0, problem.input_box, rng)

    if options["threshold_strategy"] == "None":
        model = gpc.Model_ConstantMeanMaternpML(
            "GP_bench_{}".format(options["problem"]),
            output_dim=problem.output_dim,
            covariance_params={"p": 2},
            rng=rng,
            box=problem.input_box
        )
    else:
        threshold_strategy_params = {
            "strategy": options["threshold_strategy"] ,
            "level": options["q_strategy_value"] ,
            "n_init": ni0,
            "task": options["task"]
        }
        if options["task"] == "levelset":
            threshold_strategy_params["t"] = options["t"]
        model = gpc.Model_ConstantMeanMaternp_reGP(
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

    # Optimization loop
    for step_ind in range(n_iterations):
        print(f"\niter {step_ind}")

        # Run a step of the algorithm
        try:
            algo.step()
            times_records.append(algo.training_time)
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

    # endfor

    # Prepare output directory
    i_output_path = os.path.join(options["output_dir"], "data_{}.npy".format(str(i)))
    i_times_path = os.path.join(options["output_dir"], "times_{}.npy".format(str(i)))

    # Save data
    np.save(i_output_path, np.hstack((algo.xi, algo.zi)))
    np.save(i_times_path, np.array(times_records))
