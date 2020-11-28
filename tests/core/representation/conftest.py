import pytest
import numpy as np
import random


@pytest.fixture(scope="module")
def one_to_two_and_square():
    x = np.linspace(1.0, 2.0, 11)
    y = x * x
    return [x, y]


@pytest.fixture(scope="module")
def example_array(n_vecs=1000, dim=30):
    return np.random.rand(n_vecs, dim)


@pytest.fixture(scope="module")
def distance_preserving_array_sequence(example_array):
    A = example_array
    # translation
    B = A + 1.0
    # dilation
    C = 3.0 * B
    # rotation of axes
    D = np.concatenate((C[:, 1:], C[:, :1]), axis=1)
    # reflection of random axes
    E = np.array([random.choice([-1, 1]) for i in range(D.shape[1])])[np.newaxis, :] * D

    return [A, B, C, D, E]


@pytest.fixture(scope="module")
def diagonal_multiplication_array_sequence(example_array):
    A = example_array
    R = np.diag(np.random.rand(dim))
    B = A @ M

    return [A, B]


@pytest.fixture(scope="module")
def random_multiplication_array_sequence(example_array):
    A = example_array
    R = np.random.rand(dim, np.random.randint(dim // 2, dim))
    B = A @ M

    return [A, B]
