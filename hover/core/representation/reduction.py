"""
???+ note "Linker data structures which tie (potentially multiple) dimensionality reducers to arrays."

    The point is to make it clear which reduction is in reference to which array.

    Icing on the cake: unify the syntax across different kinds of reducers.
"""
import numpy as np
from hover.core import Loggable
from .local_config import KWARG_TRANSLATOR


class DimensionalityReducer(Loggable):
    def __init__(self, array):
        """
        ???+ note "Link self to the shared input array for reduction methods."
            | Param   | Type         | Description                   |
            | :------ | :----------- | :---------------------------- |
            | `array` | `np.ndarray` | the input array to fit on     |
        """
        self.reference_array = array

    @staticmethod
    def create_reducer(method, *args, **kwargs):
        """
        ???+ note "Handle kwarg translation and dynamic imports."

            | Param      | Type   | Description              |
            | :--------- | :----- | :----------------------- |
            | `method`   | `str`  | `"umap"` or `"ivis"`     |
            | `*args`    |        | forwarded to the reducer |
            | `**kwargs` |        | translated and forwarded |
        """
        if method == "umap":
            import umap

            reducer_cls = umap.UMAP
        elif method == "ivis":
            import ivis

            reducer_cls = ivis.Ivis
        else:
            raise ValueError("Expected 'umap' or 'ivis' as reduction method")

        translated_kwargs = kwargs.copy()
        for _key, _value in kwargs.items():
            _trans_dict = KWARG_TRANSLATOR.get(_key, {})
            if method in _trans_dict:
                _trans_key = _trans_dict[method]
                translated_kwargs.pop(_key)
                translated_kwargs[_trans_key] = _value

        reducer = reducer_cls(*args, **translated_kwargs)
        return reducer

    def fit_transform(self, method, *args, **kwargs):
        """
        ???+ note "Fit and transform an array and store the reducer."
            | Param      | Type   | Description              |
            | :--------- | :----- | :----------------------- |
            | `method`   | `str`  | `"umap"` or `"ivis"`     |
            | `*args`    |        | forwarded to the reducer |
            | `**kwargs` |        | forwarded to the reducer |
        """
        reducer = DimensionalityReducer.create_reducer(method, *args, **kwargs)
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
        assert isinstance(array, np.ndarray), f"Expected np.ndarray, got {type(array)}"
        # edge case: array is too small
        if array.shape[0] < 1:
            return np.array([])

        reducer = getattr(self, method)
        return reducer.transform(array)
