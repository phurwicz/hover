"""
Manifold similarity measures for any collection of sequences of vectors.
Can be useful for improved interpretability of neural nets.
"""
from tqdm import tqdm
from scipy.spatial import procrustes
from hover.core import Loggable
from .reduction import DimensionalityReducer
from .local_config import DEFAULT_REDUCTION_METHOD


class LayerwiseManifold(Loggable):
    """
    Takes a sequence of arrays (each row of the array is a vector) and does the following:
        (1) unfold vectors into lower dimensions, typically 2D or 3D;
        (2) for every array:
            run Procrustes analysis for fitting to the previous array. The first array is fitted to itself.
    """

    DEFAULT_UNFOLD_KWARGS = {
        "umap": {
            "random_state": 0,
            "transform_seed": 0,
        }
    }

    def __init__(self, seq_arr):
        """
        :param seq_arr: sequence of arrays to fit the manifold with.
        :type seq_arr: list of numpy.ndarrays.
        """
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

    def unfold(self, method=None, **kwargs):
        """
        Compute lower-dimensional manifolds.
        :param method: the dimensionality reduction method to use.
        :type method: str
        """
        if method is None:
            method = DEFAULT_REDUCTION_METHOD

        # default kwargs should fix random state and seed
        # so that randomness does not introduce disparity
        use_kwargs = self.__class__.DEFAULT_UNFOLD_KWARGS.get(method, {}).copy()
        use_kwargs.update(kwargs)
        self.manifolds = []
        self._info(f"Running {method}...")
        for _arr in tqdm(self.arrays, total=len(self.arrays)):
            _reducer = DimensionalityReducer(_arr)
            _manifold = _reducer.fit_transform(method, **use_kwargs)
            self.manifolds.append(_manifold)
        self._good("unfolded arrays into manifolds")

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

        self._good("carried out Procrustes analysis")
        return fit_arrays, disparities
