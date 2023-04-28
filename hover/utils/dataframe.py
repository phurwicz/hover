"""
Dataframe-specific operations.
This module is intended to capture pandas/polars logic.
"""

import numpy as np
import pandas as pd
import polars as pl
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from functools import wraps


TYPE_TO_POLARS = {
    int: pl.Int64,
    float: pl.Float64,
    str: pl.Utf8,
    bool: pl.Boolean,
    np.int64: pl.Int64,
    np.float64: pl.Float64,
    pd.Int64Dtype: pl.Int64,
    pd.Float64Dtype: pl.Float64,
    pd.StringDtype: pl.Utf8,
    pd.BooleanDtype: pl.Boolean,
}


def sametype(func):
    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        value = func(obj, *args, **kwargs)
        if not isinstance(value, obj.__class__):
            value = obj.__class__(value)
        return value

    return wrapper


def convert_indices_to_list(indices, size):
    if isinstance(indices, list):
        return indices
    elif isinstance(indices, np.ndarray):
        return indices.astype(int).tolist()
    elif isinstance(indices, slice):
        assert isinstance(
            size, int
        ), f"size must be provided for slice indices, got {size}."
        return list(range(*indices.indices(size)))
    else:
        try:
            return list(indices)
        except Exception:
            raise NotImplementedError(f"Indices type {type(indices)} is not supported.")


def array_length_check(array, target_length):
    if hasattr(array, "__len__"):
        assert (
            len(array) == target_length
        ), f"length mismatch: {len(array)} != {target_length}"
    if hasattr(array, "shape"):
        assert (
            array.shape[0] == target_length
        ), f"length mismatch: {array.shape[0]} != {target_length}"


class AbstractDataframe(ABC):
    """
    ???+ note "An abstract class for hover-specific dataframe operations."
    """

    DF_TYPE = pd.DataFrame

    def __init__(self, df):
        assert isinstance(df, self.DF_TYPE), f"df must be of type {self.DF_TYPE}"
        self._df = df

    def __call__(self):
        return self._df

    @classmethod
    def construct(cls, *args, **kwargs):
        df = cls.DF_TYPE(*args, **kwargs)
        return cls(df)

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, str):
            return self._df[key]
        elif hasattr(key, "__iter__"):
            if isinstance(key[0], str):
                return self.select_columns(key)
            elif isinstance(key[0], int):
                return self.select_rows(key)
            else:
                raise NotImplementedError(f"key {key} is not supported.")
        elif isinstance(key, slice):
            return self.select_rows(range(*key.indices(self.shape[0])))
        else:
            raise NotImplementedError(f"key {key} is not supported.")

    @property
    def columns(self):
        return self._df.columns

    @columns.setter
    def columns(self):
        raise NotImplementedError("setting columns is forbidden.")

    @property
    def shape(self):
        return self._df.shape

    def column_counter(self, column):
        return Counter(self.__class__.series_tolist(self._df[column]))

    def select_columns(self, columns):
        return self.__class__(self._df[columns])

    @classmethod
    def empty_with_columns(cls, column_to_default):
        raise NotImplementedError

    @classmethod
    def concat_rows(cls, df_list):
        raise NotImplementedError

    @classmethod
    def series_values(cls, series):
        raise NotImplementedError

    @classmethod
    def series_tolist(cls, series):
        raise NotImplementedError

    @classmethod
    def series_to(cls, series, form):
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def to_pandas(self):
        raise NotImplementedError

    @abstractmethod
    def to_dict_of_lists(self):
        raise NotImplementedError

    @abstractmethod
    def to_list_of_dicts(self):
        raise NotImplementedError

    @abstractmethod
    def get_row_as_dict(self, index):
        raise NotImplementedError

    @abstractmethod
    def select_rows(self, indices):
        raise NotImplementedError

    @abstractmethod
    def filter_rows_by_operator(self, column, operator, value):
        raise NotImplementedError

    @abstractmethod
    def unique(self):
        raise NotImplementedError

    @abstractmethod
    def set_column_by_constant(self, column, value, indices=None):
        raise NotImplementedError

    @abstractmethod
    def set_column_by_array(self, column, values, indices=None):
        raise NotImplementedError

    @abstractmethod
    def column_map(self, column, mapping, indices=None, as_column=None, form="numpy"):
        raise NotImplementedError

    @abstractmethod
    def column_isin(self, column, lookup, indices=None, as_column=None, form="numpy"):
        raise NotImplementedError

    @abstractmethod
    def column_apply(
        self, column, function, indices=None, as_column=None, form="numpy"
    ):
        raise NotImplementedError

    @abstractmethod
    def row_apply(self, function, indices=None, as_column=None, form="numpy"):
        raise NotImplementedError

    @abstractmethod
    def get_cell_by_row_column(self, row_idx, column_name):
        raise NotImplementedError

    @abstractmethod
    def set_cell_by_row_column(self, row_idx, column_name):
        raise NotImplementedError


class PandasDataframe(AbstractDataframe):
    """
    ???+ note "A class for hover-specific pandas dataframe operations."
    """

    DF_TYPE = pd.DataFrame

    @classmethod
    def empty_with_columns(cls, column_to_type):
        return cls(pd.DataFrame(columns=column_to_type.keys()))

    @classmethod
    def concat_rows(cls, df_list):
        for _df in df_list:
            assert isinstance(_df, cls), f"df must be of type {cls}"
        pd_list = [df() for df in df_list]
        return cls(pd.concat(pd_list, axis=0, sort=False, ignore_index=True))

    @classmethod
    def series_values(cls, series):
        return series.values

    @classmethod
    def series_tolist(cls, series):
        return series.tolist()

    @classmethod
    def series_to(cls, series, form):
        if form == "numpy":
            return series.values
        elif form == "list":
            return series.tolist()
        elif form == "series":
            return series
        else:
            raise ValueError(f"form must be 'numpy', 'list', or 'series', got {form}")

    @sametype
    def copy(self):
        return self._df.copy()

    def to_pandas(self):
        return self._df.copy()

    def to_dict_of_lists(self):
        return self._df.to_dict(orient="list")

    def to_list_of_dicts(self):
        return self._df.to_dict(orient="records")

    def get_row_as_dict(self, index):
        assert isinstance(index, int), f"index must be int, not {type(index)}"
        return self._df.iloc[index].to_dict()

    @sametype
    def select_rows(self, indices):
        if indices is None:
            return self
        indices = convert_indices_to_list(indices, self.shape[0])
        if len(indices) == 0:
            return pd.DataFrame(columns=self.columns)
        return self._df.iloc[indices]

    @sametype
    def filter_rows_by_operator(self, column, operator, value):
        mask = operator(self._df[column], value)
        return self._df[mask].reset_index(drop=True)

    @sametype
    def unique(self, subset, keep):
        return self._df.drop_duplicates(subset, keep=keep).reset_index(drop=True)

    def set_column_by_constant(self, column, value, indices=None):
        assert np.isscalar(value), f"value must be scalar, not {type(value)}"

        if indices is None:
            self._df[column] = value
        else:
            # use conversion to avoid pandas loc taking inclusive slice
            indices = convert_indices_to_list(indices, self.shape[0])
            self._df.loc[indices, column] = value

    def set_column_by_array(self, column, values, indices=None):
        assert not np.isscalar(values), f"values must be array-like, not {type(values)}"
        if indices is None:
            target_length = self._df.shape[0]
        else:
            # use conversion to avoid pandas loc taking inclusive slice
            indices = convert_indices_to_list(indices, self.shape[0])
            target_length = len(indices)

        array_length_check(values, target_length)

        if indices is None:
            self._df[column] = values
        else:
            # use conversion to avoid pandas loc taking inclusive slice
            indices = convert_indices_to_list(indices, self.shape[0])
            self._df.loc[indices, column] = values

    def _pre_apply(self, indices, as_column):
        if indices is None:
            if as_column is not None:
                assert isinstance(
                    as_column, str
                ), f"as_column must be str, got {type(as_column)}"
            return self._df, as_column
        else:
            assert (
                as_column is None
            ), f"as_column must be None when indices are specifed, got {as_column}"
            # unlike loc, iloc needs no conversion
            return self._df.iloc[indices], None

    def _post_apply(self, series, as_column, form):
        if as_column is None:
            return self.__class__.series_to(series, form)
        else:
            self._df[as_column] = series
            return

    def column_map(self, column, mapping, indices=None, as_column=None, form="numpy"):
        subject, as_column = self._pre_apply(indices, as_column)
        series = subject[column].map(mapping)
        return self._post_apply(series, as_column, form)

    def column_isin(self, column, lookup, indices=None, as_column=None, form="numpy"):
        subject, as_column = self._pre_apply(indices, as_column)
        series = subject[column].isin(lookup)
        return self._post_apply(series, as_column, form)

    def column_apply(
        self, column, function, indices=None, as_column=None, form="numpy"
    ):
        subject, as_column = self._pre_apply(indices, as_column)
        series = subject[column].apply(function)
        return self._post_apply(series, as_column, form)

    def row_apply(self, function, indices=None, as_column=None, form="numpy"):
        subject, as_column = self._pre_apply(indices, as_column)
        series = subject.apply(function, axis=1)
        return self._post_apply(series, as_column, form)

    def get_cell_by_row_column(self, row_idx, column_name):
        return self._df.at[row_idx, column_name]

    def set_cell_by_row_column(self, row_idx, column_name, value):
        self._df.at[row_idx, column_name] = value


class PolarsDataframe(AbstractDataframe):
    """
    ???+ note "A class for hover-specific polars dataframe operations."
    """

    DF_TYPE = pl.DataFrame

    @classmethod
    def empty_with_columns(cls, column_to_type):
        return cls(
            pl.DataFrame(
                schema={
                    col: TYPE_TO_POLARS[_type] for col, _type in column_to_type.items()
                },
            )
        )

    @classmethod
    def concat_rows(cls, df_list):
        schema = None
        pl_list = []

        # basic type, length, and schema checks; get the union'ed schema
        for _df in df_list:
            assert isinstance(_df, cls), f"df must be of type {cls}"
            _pl = _df()
            if _pl.shape[0] == 0:
                continue
            if schema is None:
                schema = OrderedDict(_pl.schema)
            else:
                for col, dtype in _pl.schema.items():
                    assert (
                        schema.get(col, dtype) == dtype
                    ), f"all dataframes must have consistent schema, got {schema} and {_pl.schema}"
                schema.update(_pl.schema)
            pl_list.append(_pl)

        assert schema is not None, "all dataframes were empty"
        return cls(pl.concat(pl_list, how="diagonal"))

    @classmethod
    def series_values(cls, series):
        return series.to_numpy()

    @classmethod
    def series_tolist(cls, series):
        return series.to_list()

    @classmethod
    def series_to(cls, series, form):
        if form == "numpy":
            return series.to_numpy()
        elif form == "list":
            return series.to_list()
        elif form == "series":
            return series
        else:
            raise ValueError(f"form must be 'numpy', 'list', or 'series', got {form}")

    @sametype
    def copy(self):
        return self._df.clone()

    def to_pandas(self):
        return self._df.to_pandas()

    def to_dict_of_lists(self):
        return self._df.to_dict(as_series=False)

    def to_list_of_dicts(self):
        return self._df.to_dicts()

    def get_row_as_dict(self, index):
        assert isinstance(index, int), f"index must be int, not {type(index)}"
        return self._df.row(index, named=True)

    @sametype
    def select_rows(self, indices):
        indices = convert_indices_to_list(indices, size=self._df.shape[0])
        if len(indices) == 0:
            return self._df.head(0)
        return self._df[indices]

    @sametype
    def filter_rows_by_operator(self, column, operator, value):
        mask = self.__class__.series_values(operator(self._df[column], value))
        indices = np.where(mask)[0]
        return self.select_rows(indices)

    @sametype
    def unique(self, subset, keep):
        return self._df.unique(subset, keep=keep, maintain_order=True)

    def set_column_by_constant(self, column, value, indices=None):
        if indices is None:
            self._df = self._df.with_columns(pl.lit(value).alias(column))
        else:
            # handle slice / array and convert to lookup
            indices = set(convert_indices_to_list(indices, size=self._df.shape[0]))

            # create a temporary index column for predicating
            tmp_index_col = "index"
            while tmp_index_col in self._df.columns:
                tmp_index_col += "_"
            tmp_df = self._df.with_columns(
                pl.arange(0, self._df.shape[0]).alias(tmp_index_col)
            )

            self._df = tmp_df.with_columns(
                pl.when(pl.col(tmp_index_col).is_in(indices))
                .then(pl.lit(value))
                .otherwise(pl.col(column))
                .alias(column)
            ).drop(tmp_index_col)

    def set_column_by_array(self, column, values, indices=None):
        if indices is None:
            self._df = self._df.with_columns(pl.Series(values).alias(column))
        else:
            indices = convert_indices_to_list(indices, size=self._df.shape[0])
            array_length_check(values, len(indices))
            lookup = dict(zip(indices, values))
            patch = pl.DataFrame(
                {
                    column: [lookup.get(i, None) for i in range(self._df.shape[0])],
                }
            )
            self._df = self._df.update(patch)

    def _pre_apply(self, indices, as_column):
        # determine the column name for the result
        if as_column is None:
            col_name = "result"
            while col_name in self._df.columns:
                col_name += "_"
        else:
            assert isinstance(
                as_column, str
            ), f"as_column must be str, got {type(as_column)}"
            assert (
                indices is None
            ), f"as_column must be None when indices are specifed, got {as_column}"
            col_name = as_column

        # determine the subject of the apply
        subject = self._df if indices is None else self.select_rows(indices)()

        return subject, col_name

    def _post_apply(self, series, as_column, form):
        if as_column is None:
            return self.__class__.series_to(series, form)
        else:
            self._df = self._df.with_columns(series.alias(as_column))
            return

    def _get_return_type(self, value):
        original_type = type(value)
        if original_type not in TYPE_TO_POLARS:
            raise TypeError(f"Unsupported return type: {original_type} for {value}")
        return TYPE_TO_POLARS[original_type]

    def column_map(self, column, mapping, indices=None, as_column=None, form="numpy"):
        subject, _ = self._pre_apply(indices, as_column)
        example_value = list(mapping.values())[0]
        dtype = self._get_return_type(example_value)
        if self.shape[0] > 0:
            series = subject[column].map_dict(mapping, return_dtype=dtype)
        else:
            series = pl.Series([], dtype=dtype)
        return self._post_apply(series, as_column, form)

    def column_isin(self, column, lookup, indices=None, as_column=None, form="numpy"):
        subject, _ = self._pre_apply(indices, as_column)
        series = subject[column].is_in(lookup)
        return self._post_apply(series, as_column, form)

    def column_apply(
        self, column, function, indices=None, as_column=None, form="numpy"
    ):
        subject, _ = self._pre_apply(indices, as_column)
        if self.shape[0] > 0:
            example_value = function(self.get_cell_by_row_column(0, column))
            dtype = self._get_return_type(example_value)
            series = subject[column].apply(function, return_dtype=dtype)
        else:
            series = pl.Series([])
        return self._post_apply(series, as_column, form)

    def row_apply(self, function, indices=None, as_column=None, form="numpy"):
        # determine the return type for df.apply
        if self.shape[0] > 0:
            example_value = function(self._df.row(0, named=True))
            dtype = self._get_return_type(example_value)
        else:
            dtype = None

        subject, col = self._pre_apply(indices, as_column)

        # handle empty subject
        if subject.shape[0] == 0:
            if as_column is None:
                return self.__class__.series_to(pl.Series([]), form)
            else:
                self._df = self._df.with_columns(pl.Series([]).alias(as_column))
                return

        # create the function to be applied
        to_apply = (
            pl.struct(self._df.columns).apply(function, return_dtype=dtype).alias(col)
        )
        # apply the function
        if as_column is None:
            series = subject.with_columns(to_apply)[col]
            return self.__class__.series_to(series, form)
        else:
            assert subject is self._df, "subject must be self._df"
            self._df = subject.with_columns(to_apply)
            return

    def get_cell_by_row_column(self, row_idx, column_name):
        return self._df.row(row_idx, named=True)[column_name]

    def set_cell_by_row_column(self, row_idx, column_name, value):
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            warnings.warn(
                "Setting a single cell with a list-like object may not yet be supported by polars."
            )
        self._df[row_idx, column_name] = value
