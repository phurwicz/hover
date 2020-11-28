"""
Trajectory interpolation for sequences of vectors.
"""
from scipy import interpolate
import numpy as np


def spline(arr_per_dim, points_per_step=1, splprep_kwargs=None):
    """
    Fit a spline and evaluate it at a specified density of points.

    - param arr_per_dim(numpy.ndarray): dim-by-points array representing the part of the curve in each dimension.

    - param points_per_step(int): number of points interpolated in between each given point on the curve.

    - param splprep_kwargs(dict): keyword arguments to the splprep() function for fitting the spline in SciPy.
    """

    # cast to array if appropriate
    if isinstance(arr_per_dim, list):
        arr_per_dim = np.array(arr_per_dim)

    assert points_per_step >= 1, "Need at least one point per step"
    splprep_kwargs = splprep_kwargs or dict()

    # check the number of given points in the curve
    num_given_points = arr_per_dim[0].shape[0]
    assert num_given_points > 1, "Need at least two points to fit a line"

    # check if two vectors are almost identical, and apply a noise in that case
    # note that we did not modify arr_per_dim in place
    # and that the noise only goes up in a greedy random-walk manner
    noise_arr = np.zeros((len(arr_per_dim), num_given_points))
    for i in range(1, num_given_points):
        prev_vec, vec = arr_per_dim[:, i - 1] + noise_arr[:, i - 1], arr_per_dim[:, i]
        while np.allclose(vec + noise_arr[:, i], prev_vec):
            noise_arr[:, i] += np.random.normal(loc=0.0, scale=1e-6, size=vec.shape)

    # reduce spline order if necessary, then fit the spline parameters
    splprep_kwargs["k"] = min(3, num_given_points - 1)
    tck, u = interpolate.splprep(arr_per_dim + noise_arr, **splprep_kwargs)

    # determine points at which the spline should be evaluated
    points_to_eval = []
    for i in range(0, u.shape[0] - 1):
        _pts = np.linspace(u[i], u[i + 1], points_per_step, endpoint=False)
        points_to_eval.append(_pts)
    points_to_eval.append([u[-1]])
    points_to_eval = np.concatenate(points_to_eval)

    traj_per_dim = interpolate.splev(points_to_eval, tck)
    return traj_per_dim


def manifold_spline(seq_arr, **kwargs):
    """
    Fit a spline to every sequence of points in a manifold.
    - param seq_arr: L-sequence of M-by-N arrays each containing vectors matched by index.
    :type seq_arr: numpy.ndarray
    """
    # L is unused
    _L, M, N = seq_arr.shape

    # this gives M-by-N-by-f(L, args)
    traj_arr = np.array(
        [
            spline(np.array([seq_arr[:, _m, _n] for _n in range(N)]), **kwargs)
            for _m in range(M)
        ]
    )

    # return f(L, args)-by-M-by-N
    traj_arr = np.swapaxes(traj_arr, 1, 2)
    traj_arr = np.swapaxes(traj_arr, 0, 1)
    return traj_arr
