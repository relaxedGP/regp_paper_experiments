## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist, pdist



def mindist(sample):
    """
    Calculate the minimum distance (separation) between any pair of points in the sample.

    Parameters
    ----------
    sample : numpy.ndarray
        Array of points in the sample.

    Returns
    -------
    float
        Minimum distance between any pair of points in the sample.
    """
    D = pdist(sample)
    mindist = np.min(D)
    return mindist


def scale(sample_standard, box):
    """
    Map a standard sample in [0, 1]^dim to the given box.

    Parameters
    ----------
    sample_standard : numpy.ndarray
        Array of points in the standard sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.

    Returns
    -------
    numpy.ndarray
        Sample points mapped to the given box.
    """
    l_bounds, u_bounds = box[0], box[1]
    sample_box = qmc.scale(sample_standard, l_bounds, u_bounds)
    return sample_box


def maximinlhs(dim, n, box, rng, max_iter=1000):
    """
    Generate a maximin Latin Hypercube Sample (LHS) within the specified box.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n : int
        Number of points in the sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.
    rng : numpy.random.Generator
        Random number generator.
    max_iter : int, optional
        Maximum number of iterations for finding the sample with the maximum minimum distance, default is 1000.

    Returns
    -------
    numpy.ndarray
        Maximin Latin Hypercube Sample within the specified box.
    """
    sampler = qmc.LatinHypercube(d=dim, optimization=None, seed=rng)

    maximindist = 0
    for i in range(max_iter):
        sample = sampler.random(n)
        d = mindist(sample)
        if d > maximindist:
            maximindist = d
            sample_maximin = sample

    sample_maximin = scale(sample_maximin, box)

    return sample_maximin
