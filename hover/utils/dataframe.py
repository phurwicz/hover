"""
Dataframe-specific operations.
This module is intended to capture pandas/polars logic.
"""

import pandas as pd
import polars as pl
from abc import ABC, abstractmethod


class AbstractDataframe(ABC):
    """
    ???+ note "An abstract class for hover-specific dataframe operations."
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def empty(self, columns):
        pass


class PandasDataframe(AbstractDataframe):
    """
    ???+ note "A class for hover-specific pandas dataframe operations."
    """

    def __init__(self, *args, **kwargs):
        self._df = pd.DataFrame(*args, **kwargs)

    @classmethod
    def with_columns_only(cls, columns):
        return cls(columns=columns)


class PolarsDataframe(AbstractDataframe):
    """
    ???+ note "A class for hover-specific polars dataframe operations."
    """

    def __init__(self, *args, **kwargs):
        self._df = pl.DataFrame(*args, **kwargs)

    @classmethod
    def with_columns_only(cls, columns):
        return cls({col: [] for col in columns})
