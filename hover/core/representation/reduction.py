"""
Linker data structure which ties (potentially multiple) dimensionality reducers to arrays.

The point is to make it clear which reduction is in reference to which array.
Icing on the cake: unify the syntax across different kinds of reducers.
"""


class DimensionalityReducer(object):
    def __init__(self, array):
        """Link self to the shared input array for reduction methods.

        :param array: the input array to be transformed.
        :type array: np.ndarray
        """
        self.reference_array = array
        self.method_error_msg = "Expected 'umap' or 'ivis' as reduction method"

    def fit_transform(self, method, *args, **kwargs):
        """Fit and transform an array and store the reducer.

        :param method: the dimensionality reduction method to use.
        :type method: str, "umap" or "ivis"
        :param *args: positional parameters for the reducer.
        :param **kwargs: keyword parameters for the reducer.
        """
        assert method in ["umap", "ivis"], self.method_error_msg
        if method == "umap":
            import umap

            reducer = umap.UMAP(*args, **kwargs)
        elif method == "ivis":
            import ivis

            reducer = ivis.Ivis(*args, **kwargs)

        embedding = reducer.fit_transform(self.reference_array)
        setattr(self, method, reducer)
        return embedding

    def transform(self, array, method):
        """Transform an array with a prepared reducer.

        :param array: the input array to be transformed.
        :type array: np.ndarray
        """
        assert method in ["umap", "ivis"], self.method_error_msg

        reducer = getattr(self, method)
        return reducer.transform(array)
