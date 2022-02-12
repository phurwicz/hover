"""
???+ note "Linker data structures which tie (potentially multiple) dimensionality reducers to arrays."

    The point is to make it clear which reduction is in reference to which array.

    Icing on the cake: unify the syntax across different kinds of reducers.
"""
import numpy as np
from hover.core import Loggable


class DimensionalityReducer(Loggable):
    AVAILABLE_METHODS = {"umap", "ivis"}
    METHOD_ERROR_MSG = "Expected 'umap' or 'ivis' as reduction method"

    def __init__(self, array):
        """
        ???+ note "Link self to the shared input array for reduction methods."
            | Param   | Type         | Description                   |
            | :------ | :----------- | :---------------------------- |
            | `array` | `np.ndarray` | the input array to fit on     |
        """
        self.reference_array = array

    def fit_transform(self, method, *args, **kwargs):
        """
        ???+ note "Fit and transform an array and store the reducer."
            | Param      | Type   | Description              |
            | :--------- | :----- | :----------------------- |
            | `method`   | `str`  | `"umap"` or `"ivis"`     |
            | `*args`    |        | forwarded to the reducer |
            | `**kwargs` |        | forwarded to the reducer |
        """
        if method == "umap":
            import umap

            reducer = umap.UMAP(*args, **kwargs)
        elif method == "ivis":
            import ivis

            reducer = ivis.Ivis(*args, **kwargs)
        else:
            raise ValueError(self.__class__.METHOD_ERROR_MSG)

        embedding = reducer.fit_transform(self.reference_array)
        setattr(self, method, reducer)
        return embedding

    def transform(self, array, method):
        """
        ???+ note "Transform an array with a already-fitted reducer."
            | Param      | Type         | Description              |
            | :--------- | :----------- | :----------------------- |
            | `array`    | `np.ndarray` | the array to transform   |
            | `method`   | `str`        | `"umap"` or `"ivis"`     |
        """
        assert method in ["umap", "ivis"], self.method_error_msg
        assert isinstance(array, np.ndarray), f"Expected np.ndarray, got {type(array)}"
        # edge case: array is too small
        if array.shape[0] < 1:
            return np.array([])

        reducer = getattr(self, method)
        return reducer.transform(array)
