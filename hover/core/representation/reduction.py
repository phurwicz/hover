"""
Linker data structure which ties (potentially multiple) dimensionality reducers to arrays.

The point is to make it clear which reduction is in reference to which array.
Icing on the cake: unify the syntax across different kinds of reducers.
"""
import numpy as np


class DimensionalityReducer(object):
    AVAILABLE_METHODS = {"umap", "ivis"}
    METHOD_ERROR_MSG = "Expected 'umap' or 'ivis' as reduction method"

    def __init__(self, array):
        """Link self to the shared input array for reduction methods.

        :param array: the input array to be transformed.
        :type array: np.ndarray
        """
        self.reference_array = array

    def fit_transform(self, method, *args, **kwargs):
        """Fit and transform an array and store the reducer.

        :param method: the dimensionality reduction method to use.
        :type method: str, "umap" or "ivis"
        :param *args: positional parameters for the reducer.
        :param **kwargs: keyword parameters for the reducer.
        """
        if method == "umap":
            try:
                import umap

                reducer = umap.UMAP(*args, **kwargs)
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Please install umap-learn via pip.")

        elif method == "ivis":
            try:
                import ivis

                reducer = ivis.Ivis(*args, **kwargs)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install ivis[cpu] or ivis[gpu] via pip."
                )
        else:
            raise ValueError(self.__class__.METHOD_ERROR_MSG)

        embedding = reducer.fit_transform(self.reference_array)
        setattr(self, method, reducer)
        return embedding

    def transform(self, array, method):
        """Transform an array with a prepared reducer.

        :param array: the input array to be transformed.
        :type array: np.ndarray
        """
        assert method in ["umap", "ivis"], self.method_error_msg
        assert isinstance(array, np.ndarray), f"Expected np.ndarray, got {type(array)}"
        # edge case: array is too small
        if array.shape[0] < 1:
            return np.array([])

        reducer = getattr(self, method)
        return reducer.transform(array)
