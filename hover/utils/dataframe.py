"""
Dataframe-specific operations.
This module is intended to capture pandas/polars logic.
"""

import numpy as np
import pandas as pd
import polars as pl
from abc import ABC, abstractmethod
from collections import Counter
from functools import wraps


def sametype(func):
    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        value = func(obj, *args, **kwargs)
        if not isinstance(value, obj.__class__):
            value = obj.__class__(value)
        return value

    return wrapper


class AbstractDataframe(ABC):
    """
    ???+ note "An abstract class for hover-specific dataframe operations."
    """

    DF_TYPE = None

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
    def empty_with_columns(cls, columns):
        raise NotImplementedError

    @classmethod
    def vertical_concat(cls, df_a, df_b):
        raise NotImplementedError

    @classmethod
    def series_values(cls, series):
        raise NotImplementedError

    @classmethod
    def series_tolist(cls, series):
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
    def row(self, index):
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
    def row_apply(self, function, indices=None):
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
    def empty_with_columns(cls, columns):
        return cls(pd.DataFrame(columns=columns))

    @classmethod
    def vertical_concat(cls, df_list):
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

    @sametype
    def copy(self):
        return self._df.copy()

    def to_pandas(self):
        return self._df.copy()

    def to_dict_of_lists(self):
        return self._df.to_dict(orient="list")

    def to_list_of_dicts(self):
        return self._df.to_dict(orient="records")

    def row(self, index):
        assert isinstance(index, int), f"index must be int, not {type(index)}"
        return self._df.iloc[index]

    @sametype
    def select_rows(self, indices):
        assert (
            isinstance(indices, list)
            or isinstance(indices, np.ndarray)
            or isinstance(indices, slice)
            or isinstance(indices, range)
        ), f"indices must be list, np.ndarray, or slice, not {type(indices)}"
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
            self._df.loc[indices, column] = value

    def set_column_by_array(self, column, values, indices=None):
        assert not np.isscalar(values), f"values must be array-like, not {type(values)}"
        target_length = self._df.shape[0] if indices is None else len(indices)
        if hasattr(values, "__len__"):
            assert (
                len(values) == target_length
            ), f"length mismatch: {len(values)} != {self._df.shape[0]}"
        if hasattr(values, "shape"):
            assert (
                values.shape[0] == target_length
            ), f"length mismatch: {values.shape[0]} != {self._df.shape[0]}"

        if indices is None:
            self._df[column] = values
        else:
            self._df.loc[indices, column] = values

    def row_apply(self, function, indices=None):
        if indices is None:
            return self._df.apply(function, axis=1)
        else:
            return self._df.iloc[indices].apply(function, axis=1)

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
    def empty_with_columns(cls, columns):
        return cls(pl.DataFrame({col: [] for col in columns}))

    @classmethod
    def vertical_concat(cls, df_list):
        for _df in df_list:
            assert isinstance(_df, cls), f"df must be of type {cls}"
        pl_list = [df() for df in df_list]
        return cls(pl.concat(pl_list, how="vertical"))

    @classmethod
    def series_values(cls, series):
        return series.to_numpy()

    @classmethod
    def series_tolist(cls, series):
        return series.to_list()

    @sametype
    def copy(self):
        return self._df.clone()

    def to_pandas(self):
        return self._df.to_pandas()

    def to_dict_of_lists(self):
        return self._df.to_dict(as_series=False)

    def to_list_of_dicts(self):
        return self._df.to_dicts()

    def row(self, index):
        assert isinstance(index, int), f"index must be int, not {type(index)}"
        return self._df[index]

    @sametype
    def select_rows(self, indices):
        return self._df[indices]

    @sametype
    def filter_rows_by_operator(self, column, operator, value):
        mask = operator(self._df[column], value)
        indices = np.where(mask)[0]
        return self._df[indices]

    @sametype
    def unique(self, subset, keep):
        return self._df.unique(subset, keep=keep, maintain_order=True)

    def set_column_by_constant(self, column, value, indices=None):
        if indices is None:
            self._df = self._df.with_columns(pl.lit(value).alias(column))
        else:
            self._df = self._df.with_columns(
                pl.when(pl.col("index").is_in(indices))
                .then(pl.lit(value))
                .otherwise(pl.col(column))
                .alias(column)
            )

    def set_column_by_array(self, column, values, indices=None):
        if indices is None:
            self._df = self._df.with_columns(pl.Series(values).alias(column))
        else:
            raise NotImplementedError(
                "Column-list partial assign is a to-do for polars."
            )

    def row_apply(self, function, indices=None):
        raise NotImplementedError("Row-wise apply is a to-do for polars.")

    def get_cell_by_row_column(self, row_idx, column_name):
        return self._df[row_idx, column_name]

    def set_cell_by_row_column(self, row_idx, column_name, value):
        self._df[row_idx, column_name] = value
