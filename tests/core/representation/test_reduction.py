from hover.core.representation.reduction import DimensionalityReducer
import numpy as np
import pytest


@pytest.mark.lite
def test_create_reducer(n_points=1000):
    # if marked as lite, only test the default reducer library
    from umap import UMAP

    reducer = DimensionalityReducer.create_reducer(
        "umap",
        dimension=4,
        n_neighbors=10,
    )
    assert isinstance(reducer, UMAP)
    # dimension is expected to override n_components (default 2)
    assert reducer.n_components == 4
    # other kwargs are expected to simply get forwarded
    assert reducer.n_neighbors == 10


def test_dimensionality_reduction(n_points=1000):

    arr = np.random.rand(n_points, 20)
    reducer = DimensionalityReducer(arr)

    reducer.fit_transform(
        "umap", n_neighbors=3, min_dist=0.01, dimension=3, metric="euclidean"
    )
    embedding = reducer.transform(arr)
    assert embedding.shape == (n_points, 3)
    embedding = reducer.transform(np.array([]))
    assert embedding.shape == (0,)

    reducer.fit_transform(
        "ivis", dimension=4, k=3, distance="pn", batch_size=16, epochs=20
    )
    embedding = reducer.transform(arr, "ivis")
    assert embedding.shape == (n_points, 4)

    try:
        reducer.fit_transform("invalid_method")
        pytest.fail("Expected exception from invalid reduction method.")
    except ValueError:
        pass
