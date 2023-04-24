from hover.utils.dataframe import PandasDataframe, PolarsDataframe
import numpy as np
import pandas as pd
import polars as pl
import pytest
import operator


SERIES_VALUE_TEST_CASES = [
    list(range(30)),
    list("abc" * 10),
    [True, False] * 15,
]

DATAFRAME_VALUE_TEST_CASES = [
    {"int": list(range(30)), "bool": [False, True] * 15, "str": list("abc" * 10)},
    {
        "int": list(range(30)),
        "bool": [True, False] * 15,
        "array": [np.array([float(i)]) for i in range(30)],
    },
]


class TestPandasDataframe:
    """Tests for the PandasDataframe class."""

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_basics(self, df_data):
        pd_df = pd.DataFrame(df_data)
        df = PandasDataframe(pd_df)
        assert df() is pd_df
        assert df().equals(PandasDataframe.construct(df_data)())
        assert df.copy()() is not df()
        assert (df.columns == df().columns).all()
        assert df.shape == df().shape

    def test_empty_with_columns(self):
        df = PandasDataframe.empty_with_columns(["a", "b"])
        pd_df = pd.DataFrame(columns=["a", "b"])
        assert df().equals(pd_df)
        assert df.shape == (0, 2)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_vertical_concat(self, df_data):
        df_a = PandasDataframe.construct(df_data)
        df_b = df_a.copy()
        pd_df_a = df_a()
        pd_df_b = df_b()
        df_ab = PandasDataframe.vertical_concat([df_a, df_b])
        pd_df_ab = pd.concat([pd_df_a, pd_df_b], axis=0, ignore_index=True)
        assert df_ab().equals(pd_df_ab)

        try:
            _ = PandasDataframe.vertical_concat([pd_df_a, pd_df_b])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass

    @pytest.mark.parametrize("values", SERIES_VALUE_TEST_CASES)
    def test_series_class_methods(self, values):
        series = pd.Series(values)
        assert np.equal(PandasDataframe.series_values(series), np.array(values)).all()
        assert PandasDataframe.series_tolist(series) == list(values)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_to_pandas(self, df_data):
        df = pd.DataFrame(df_data)
        pd_export = PandasDataframe(df).to_pandas()
        assert isinstance(pd_export, pd.DataFrame)
        assert pd_export.equals(df)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_to_dict_of_lists(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        assert df.to_dict_of_lists() == pd_df.to_dict(orient="list")

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_to_list_of_dicts(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        assert df.to_list_of_dicts() == pd_df.to_dict(orient="records")

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_row(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        row_from_call = df().iloc[0]
        row_from_pd = pd_df.iloc[0]
        assert (df.row(0) == row_from_call).all()
        assert (df.row(0) == row_from_pd).all()

        try:
            _ = df.row([0, 1])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_select_rows(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        for _arg in [
            [0, 1],
            np.array([0, 1, 3]),
            slice(0, 10, 2),
            range(0, 10, 2),
        ]:
            assert df.select_rows(_arg)().equals(pd_df.iloc[_arg])

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_filter_rows_by_operator(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        for _op in [operator.eq, operator.ne, operator.gt, operator.lt]:
            _slice_a = df.filter_rows_by_operator("int", _op, 5)()
            _slice_b = pd_df[pd_df["int"].apply(lambda x: _op(x, 5))].reset_index(
                drop=True
            )
            assert _slice_a.equals(_slice_b)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_unique(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        unique_a = df.unique("bool", keep="last")()
        unique_b = pd_df.drop_duplicates("bool", keep="last").reset_index(drop=True)

        assert unique_a.equals(unique_b)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_set_column_by_constant(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        col, value = df.columns[0], df.row(0)[df.columns[0]]
        df.set_column_by_constant(col, value)
        pd_df[col] = value
        assert df().equals(pd_df)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_set_column_by_array(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        col, indices = df.columns[0], list(range(0, 10, 2))
        values = df.select_rows(indices)[col]
        df.set_column_by_array("int", values[::-1], indices=indices)
        pd_df.loc[indices, "int"] = values[::-1]
        assert df().equals(pd_df)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_row_apply(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        indices = list(range(0, 10, 2))

        def func(row):
            return str(row["int"])

        assert df.row_apply(func).equals(pd_df.apply(func, axis=1))
        assert df.row_apply(func, indices=indices).equals(
            pd_df.loc[indices].apply(func, axis=1)
        )

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_get_cell_by_row_column(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        row = np.random.randint(0, df.shape[0])
        col = np.random.choice(df.columns)
        assert df.get_cell_by_row_column(row, col) == pd_df.at[row, col]

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_set_cell_by_row_column(self, df_data):
        df = PandasDataframe.construct(df_data)
        pd_df = pd.DataFrame(df_data)
        row = np.random.randint(0, df.shape[0])
        col = np.random.choice(df.columns)
        old_value = df.get_cell_by_row_column(row, col)
        value = df.get_cell_by_row_column(df.shape[0] - row, col)
        df.set_cell_by_row_column(row, col, value)
        assert old_value == pd_df.at[row, col]
        pd_df.at[row, col] = value
        assert df.get_cell_by_row_column(row, col) == pd_df.at[row, col]


class TestPolarsDataframe:
    """Tests for the PandasDataframe class."""

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_basics(self, df_data):
        pl_df = pl.DataFrame(df_data)
        df = PolarsDataframe(pl_df)
        assert df() is pl_df
        assert df().frame_equal(PolarsDataframe.construct(df_data)())
        assert df.copy()() is not df()
        assert df.columns == df().columns
        assert df.shape == df().shape

    def test_empty_with_columns(self):
        df = PolarsDataframe.empty_with_columns(["a", "b"])
        pl_df = pl.DataFrame({_col: [] for _col in ["a", "b"]})
        assert df().frame_equal(pl_df)
        assert df.shape == (0, 2)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_vertical_concat(self, df_data):
        df_a = PolarsDataframe.construct(df_data)
        df_b = df_a.copy()
        pl_df_a = df_a()
        pl_df_b = df_b()
        df_ab = PolarsDataframe.vertical_concat([df_a, df_b])
        pl_df_ab = pl.concat([pl_df_a, pl_df_b], how="vertical")
        pd_df_ab = pd.concat(
            [pl_df_a.to_pandas(), pl_df_b.to_pandas()],
            axis=0,
            ignore_index=True,
        )
        assert df_ab().frame_equal(pl_df_ab)
        assert df_ab().to_pandas().equals(pd_df_ab)

        try:
            _ = PolarsDataframe.vertical_concat([pl_df_a, pl_df_b])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass
