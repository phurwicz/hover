"""
Manifold similarity measures for any collection of sequences of vectors.
Can be useful for improved interpretability of neural nets.
"""
from .reduction import DimensionalityReducer
from hover import module_params
from tqdm import tqdm
import numpy as np
from scipy.spatial import procrustes

DEFAULT_UMAP_PARAMS = {
    "n_components": 2,
    "n_neighbors": 30,
    "min_dist": 0.1,
    "metric": "euclidean",
    "random_state": 0,
    "transform_seed": 0,
}


class LayerwiseManifold(object):
    """
    Takes a sequence of arrays (each row of the array is a vector) and does the following:
        (1) unfold vectors into lower dimensions, typically 2D or 3D;
        (2) for every array:
            run Procrustes analysis for fitting to the previous array. The first array is fitted to itself.
    """

    def __init__(self, seq_arr):
        """
        :param seq_arr: sequence of arrays to fit the manifold with.
        :type seq_arr: list of numpy.ndarrays.
        """
        self.logger = module_params.default_logger()
        self.arrays = seq_arr[:]
        self.validate()
        self.standardize()

    def validate(self):
        """
        Sanity check of array dimensions.
        """
        assert (
            len(self.arrays) > 1
        ), "Need at least two arrays to compute layerwise manifold."
        self.n_vecs = self.arrays[0].shape[0]
        for _arr in self.arrays:
            assert _arr.shape[0] == self.n_vecs
        self.logger.good("Validated dimensions of input arrays")

    def standardize(self):
        """
        Standardize each array to the Procrustes form where
            - tr(A^T A) = 1
            - A.mean(axis=0) = 0
        """

        def transform(arr):
            matrix, _, _ = procrustes(arr, arr)
            return matrix

        self.arrays = [transform(_arr) for _arr in self.arrays]
        self.logger.good("Standardized input arrays")

    def unfold(self, method="umap", reducer_kwargs=DEFAULT_UMAP_PARAMS):
        """
        Compute lower-dimensional manifolds using UMAP.
        :param method: the dimensionality reduction method to use.
        :type method: str
        """
        import umap

        self.manifolds = []
        self.logger.info(f"Running {method}...")
        for _arr in tqdm(self.arrays, total=len(self.arrays)):
            _reducer = DimensionalityReducer(_arr)
            _manifold = _reducer.fit_transform(method, **reducer_kwargs)
            self.manifolds.append(_manifold)
        self.logger.good("Successfully unfolded arrays into manifolds")

    def procrustes(self, arrays=None):
        """
        Run Procrustes analysis, optionally on a specified list of arrays.
        """
        if arrays is None:
            arrays = self.manifolds
        disparities = []
        fit_arrays = []

        # fit each array to its fitted predecessor
        for i, _arr in enumerate(arrays):
            if i == 0:
                # fit the first array to itself
                _, _matrix, _disparity = procrustes(_arr, _arr)
            else:
                _, _matrix, _disparity = procrustes(fit_arrays[i - 1], _arr)
            disparities.append(_disparity)
            fit_arrays.append(_matrix)

        self.logger.good("Successfully carried out Procrustes analysis")
        return fit_arrays, disparities
