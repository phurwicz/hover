from hover.core.representation.reduction import DimensionalityReducer
import numpy as np


def test_dimensionality_reduction(n_points=1000):

    arr = np.random.rand(n_points, 20)
    reducer = DimensionalityReducer(arr)

    reducer.fit_transform(
        "umap", n_neighbors=3, min_dist=0.01, n_components=2, metric="euclidean"
    )
    embedding = reducer.transform(arr, "umap")
    assert embedding.shape == (n_points, 2)

    reducer.fit_transform(
        "ivis", embedding_dims=2, k=3, distance="pn", margin=1, batch_size=16, epochs=20
    )
    embedding = reducer.transform(arr, "ivis")
    assert embedding.shape == (n_points, 2)
