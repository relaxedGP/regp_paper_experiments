import numpy as np
import sys, os
import matplotlib.pyplot as plt
import gpmpcontrib.levelset.test_problems as test_problems

input_folder = sys.argv[1]
test_function = sys.argv[2]
regp_method = sys.argv[3]
step = int(sys.argv[4])

seq_method = "straddle"

n_runs = 30

problem = getattr(test_problems, test_function)

d = len(problem.input_box[0])

path = os.path.join(input_folder, seq_method, test_function, regp_method)

full_data = []

for i in range(n_runs):
    zi_relaxed = np.load(os.path.join(path, "logs_{}".format(i), "zi_relaxed_{}.npy".format(step))).ravel()
    zi = np.load(os.path.join(path, "data_{}.npy".format(i)))[:, -1][:(10 * d + step + 1)]

    plt.figure()
    plt.plot(zi, zi_relaxed, 'o', alpha=0.3)

    plt.show()