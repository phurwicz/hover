from hover.core.representation.trajectory import spline, manifold_spline
import numpy as np


def test_spline(one_to_two_and_square):
    x, y = one_to_two_and_square

    traj_x, traj_y = spline([x, y], points_per_step=1, splprep_kwargs={"k": 2})
    assert (np.absolute(traj_x - x) < 1e-2).all()
    assert (np.absolute(traj_y - y) < 1e-2).all()


def test_manifold_spline(one_to_two_and_square, num_points=100):
    # shape: dim-by-step
    arr = np.array(one_to_two_and_square)

    # shape: point-by-dim-by-step
    arr = np.array([arr] * num_points)

    # shape: step-by-point-by-dim
    arr = np.swapaxes(arr, 1, 2)
    arr = np.swapaxes(arr, 0, 1)
    L, M, N = arr.shape

    # add a displacement that varies by point
    arr += np.linspace(0.0, 0.1, num_points)[np.newaxis, :, np.newaxis]

    traj = manifold_spline(arr, points_per_step=1, splprep_kwargs={"k": 2})
    assert traj.shape == (L, M, N)

    traj = manifold_spline(arr, points_per_step=3, splprep_kwargs={"k": 2})
    assert traj.shape == (3 * L - 2, M, N)
