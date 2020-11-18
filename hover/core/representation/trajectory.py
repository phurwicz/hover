"""
Trajectory interpolation for sequences of vectors.
"""
from scipy import interpolate
import numpy as np
import numba


def spline(arr_per_dim, points_per_step=1, splprep_kwargs={}):
    """
    Fit a spline and evaluate it at a specified density of points.
    :param arr_per_dim: arrays representing the part of the curve in each dimension.
    :type arr_per_dim: list of numpy.array
    :param points_per_step: number of points interpolated in between each given point on the curve.
    :type points_per_step: int
    :param splprep_kwargs: keyword arguments to the splprep() function for fitting the spline in SciPy.
    :type splprep_kwargs: dict
    """
    assert points_per_step >= 1, "Need at least one point per step"

    # check the number of given points in the curve
    num_given_points = arr_per_dim[0].shape[0]
    assert num_given_points > 1, "Need at least two points to fit a line"

    # reduce spline order if necessary, then fit the spline parameters
    splprep_kwargs["k"] = min(3, num_given_points - 1)
    tck, u = interpolate.splprep(arr_per_dim, **splprep_kwargs)

    # determine points at which the spline should be evaluated
    points_to_eval = []
    for i in range(0, u.shape[0] - 1):
        _pts = np.linspace(u[i], u[i + 1], points_per_step, endpoint=False)
        points_to_eval.append(_pts)
    points_to_eval.append([u[-1]])
    points_to_eval = np.concatenate(points_to_eval)

    traj_per_dim = interpolate.splev(points_to_eval, tck)
    return traj_per_dim


def manifold_spline(seq_arr, points_per_step=1, splprep_kwargs={}):
    """
    Fit a spline to every sequence of points in a manifold.
    :param seq_arr: L-sequence of M-by-N arrays each containing vectors matched by index.
    :type seq_arr: numpy.ndarray
    :param points_per_step: number of points interpolated in between each given point on the curve.
    :type points_per_step: int
    :param splprep_kwargs: keyword arguments to the splprep() function for fitting the spline in SciPy.
    :type splprep_kwargs: dict
    """
    L, M, N = seq_arr.shape

    # this gives M-by-N-by-f(L, args)
    traj_arr = np.array(
        [
            spline(
                [seq_arr[:, _m, _n] for _n in range(N)], points_per_step, splprep_kwargs
            )
            for _m in range(M)
        ]
    )

    # return f(L, args)-by-M-by-N
    traj_arr = np.swapaxes(traj_arr, 1, 2)
    traj_arr = np.swapaxes(traj_arr, 0, 1)
    return traj_arr
