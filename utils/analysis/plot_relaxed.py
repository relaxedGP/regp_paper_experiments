import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import numpy as np
import matplotlib.pyplot as plt
import gpmpcontrib.levelset.test_problems as test_problems
import gpmpcontrib as gpc

import sys, os

output_path = sys.argv[1]

np.random.seed(0)

threshold = True

problem_name = 'g10c6mod'

if problem_name == 'g10c6':
    title = 'c6'
elif problem_name == 'g10c6mod':
    title = 'c6'
elif problem_name == 'g10c6modmod':
    title = 'c6'
else:
    raise ValueError(problem_name)

problem = getattr(test_problems, problem_name)

box = problem.input_box

#X = stats.qmc.scale(np.random.uniform(size=(100, problem.input_dim)), box[0], box[1])
X = np.load("C:/Users/PETIT/data/final_levelset/results/straddle/g10c6mod/Concentration/data_0.npy")[:, :8]
y = problem.eval(X).ravel()

min_abs = np.abs(y).min()/2

# Build model

threshold_strategy_params = {
    "strategy": "Concentration",
    "level": 0.25,
    "task": "levelset",
    "t": 0.0
}

crit_optim_options = {
    "method": "SLSQP",
    "relaxed_init":  "quad_prog",
    "ftol": 1e-06,
    "eps": 1e-08,
    "maxiter": 15000
}

model = gpc.Model_ConstantMeanMaternp_reGP(
    threshold_strategy_params,
    "reGP_model",
    crit_optim_options=crit_optim_options,
    output_dim=problem.output_dim,
    covariance_params={"p": 2},
    rng=np.random.default_rng(),
    box=problem.input_box
)

model.select_params(X, y, force_param_initial_guess=True)

########

mult = 1#/1E6

y_slack = model.predict(X, y, X)[0].ravel()

plt.figure(figsize=(6 * 0.7, 4 * 0.7))

#plt.title(title)

lower_threshold = model.models[0]["R"][0][1]
upper_threshold = model.models[0]["R"][1][0]

key_relaxed = (y <= lower_threshold) | (upper_threshold <= y)
print("Prop relaxed: ", key_relaxed.mean())

plt.plot(y[key_relaxed] * mult, y_slack[key_relaxed] * mult, 'bo')
plt.plot(y[~key_relaxed] * mult, y_slack[~key_relaxed] * mult, 'ks')

plt.axhline(0, color='green', linewidth=0.5)
plt.axvline(0, color='green', linewidth=0.5)

# plt.axhline(lower_threshold * mult, color='k', linewidth=0.5)
# plt.axhline(upper_threshold * mult, color='k', linewidth=0.5)
plt.axvline(lower_threshold * mult, color='k', label='THRESHOLDS', linewidth=0.5)
plt.axvline(upper_threshold * mult, color='k', linewidth=0.5)

##
# output_reg = y_slack * mult
# input_reg = y * mult
#
# regressor = input_reg ** 7
#
# beta = (regressor * output_reg).sum()/(regressor * regressor).sum()
#
# input_pred = np.linspace(input_reg.min(), input_reg.max(), 1000).reshape(-1, 1)
# output_pred = beta * input_pred ** 7
#
# plt.plot(input_pred, output_pred)

##

#plt.legend()

plt.xlabel("FUNC")
plt.ylabel("PRED")

#
plt.yticks(
    [],
    []
)

plt.xticks(
    [lower_threshold, 0, upper_threshold],
    [r't10', 't12', r't11']
)
#

plt.text(12, 3800, "RSET", va='center')
plt.text(-14, 3800, "RSET", va='center')
plt.text(-1, 3800, "GSET", va='center')

# plt.text(19, 50, "RSET", rotation=270, va='center')
# plt.text(19, -50, "RSET", rotation=270, va='center')
# plt.text(19, -10, "GSET", rotation=270, va='center')

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_path, "{}_slack.pgf".format(problem_name.lower())))
plt.close()
