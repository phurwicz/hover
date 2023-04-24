"""
Dataframe-specific operations.
This module is intended to capture pandas/polars logic.
"""

import numpy as np
import pandas as pd
import polars as pl
from abc import ABC, abstractmethod
from collections import Counter


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

    def copy(self):
        return self.__class__(self._df.copy())

    def __getitem__(self, key):
        return self._df[key]

    @property
    def columns(self):
        return self._df.columns

    @property
    def shape(self):
        return self._df.shape

    def column_counter(self, column):
        return Counter(self.__class__.series_tolist(self._df[column]))

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
    def to_pandas(self):
        raise NotImplementedError

    @abstractmethod
    def to_dict_of_lists(self):
        raise NotImplementedError

    @abstractmethod
    def to_list_of_dicts(self):
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
    def column_assign_constant(self, column, value, indices=None):
        raise NotImplementedError

    @abstractmethod
    def column_assign_list(self, column, values, indices=None):
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
        return cls(columns=columns)

    @classmethod
    def vertical_concat(cls, df_list):
        pd_list = [df() for df in df_list]
        return cls(pd.concat(pd_list, axis=0, sort=False, ignore_index=True))

    @classmethod
    def series_values(cls, series):
        return series.values

    @classmethod
    def series_tolist(cls, series):
        return series.tolist()

    def to_pandas(self):
        return self._df.copy()

    def to_dict_of_lists(self):
        return self._df.to_dict(orient="list")

    def to_list_of_dicts(self):
        return self._df.to_dict(orient="records")

    def select_rows(self, indices):
        return self._df.iloc[indices]

    def filter_rows_by_operator(self, column, operator, value):
        mask = operator(self._df[column], value)
        return self.__class__(self._df[mask].reset_index(drop=True))

    def unique(self, subset, keep):
        return self.__class__(
            self._df.drop_duplicates(subset, keep=keep).reset_index(drop=True)
        )

    def column_assign_constant(self, column, value, indices=None):
        if indices is None:
            self._df[column] = value
        else:
            self._df.loc[indices, column] = value

    def column_assign_list(self, column, values, indices=None):
        if indices is None:
            self._df[column] = values
        else:
            self._df.loc[indices, column] = values

    def row_apply(self, function, indices=None):
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
        return cls({col: [] for col in columns})

    @classmethod
    def vertical_concat(cls, df_list):
        pl_list = [df() for df in df_list]
        return cls(pl.concat(pl_list, how="vertical"))

    @classmethod
    def series_values(cls, series):
        return series.to_numpy()

    @classmethod
    def series_tolist(cls, series):
        return series.to_list()

    def to_pandas(self):
        return self._df.to_pandas()

    def to_dict_of_lists(self):
        return self._df.to_dict(as_series=False)

    def to_list_of_dicts(self):
        return self._df.to_dicts()

    def select_rows(self, indices):
        return self._df[indices]

    def filter_rows_by_operator(self, column, operator, value):
        mask = operator(self._df[column], value)
        indices = np.where(mask)[0]
        return self.__class__(self._df[indices])

    def unique(self, subset, keep):
        return self.__class__(self._df.unique(subset, keep=keep, maintain_order=True))

    def column_assign_constant(self, column, value, indices=None):
        if indices is None:
            self._df = self._df.with_column(pl.lit(value).alias(column))
        else:
            self._df = self._df.with_column(
                pl.when(pl.col("index").is_in(indices))
                .then(pl.lit(value))
                .otherwise(pl.col(column))
                .alias(column)
            )

    def column_assign_list(self, column, values, indices=None):
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
