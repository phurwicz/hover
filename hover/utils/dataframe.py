"""
Dataframe-specific operations.
This module is intended to capture pandas/polars logic.
"""

import pandas as pd


def dataframe_with_no_rows(columns):
    return pd.DataFrame(columns=columns)
