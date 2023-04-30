from hover.utils.dataframe import (
    PandasDataframe,
    PolarsDataframe,
    convert_indices_to_list,
    TYPE_TO_POLARS,
)
from functools import lru_cache
from pprint import pformat
import gc
import numpy as np
import pandas as pd
import polars as pl
import time
import faker
import pytest
import operator


EN_FAKER = faker.Faker("en")

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

HASHABLE_COLUMNS = ["int", "bool", "str"]

ROW_INDICES_TEST_CASES = [
    [0, 1],
    [],
    np.array([0, 1, 3]),
    slice(0, 10, 2),
    range(0, 10, 2),
    None,
]


@lru_cache(maxsize=10)
def generate_benchmark_dataframe(size, types=("int", "str"), lazy=True):
    data = dict()
    for _type in types:
        if _type == "int":
            data[_type] = np.random.randint(0, 100, size=size)
        elif _type == "str":
            data[_type] = (
                [EN_FAKER.paragraph(1)] * size
                if lazy
                else [EN_FAKER.paragraph(1) for _ in range(size)]
            )
        else:
            raise ValueError(f"Unrecognized type: {_type}")
    return data


def numpy_to_native(value):
    """
    Convert numpy types to native types.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    else:
        assert isinstance(value, (list, tuple, dict, str, bool, int, float))
    return value


def get_dataframes(df_data):
    """
    Subroutine for creating dataframes in tests.
    """
    df_pd = PandasDataframe.construct(df_data)
    df_pl = PolarsDataframe.construct(df_data)
    pd_df = pd.DataFrame(df_data)
    pl_df = pl.DataFrame(df_data)

    return df_pd, df_pl, pd_df, pl_df


def assert_equivalent_dataframes(df_pd, df_pl, pd_df, pl_df):
    """
    Subroutine for checking dataframe values.
    """
    assert df_pd.equals(pd_df), f"{pformat(df_pd)}\n{pformat(pd_df)}"
    assert df_pl.frame_equal(pl_df), f"{pformat(df_pl)}\n{pformat(pl_df)}"
    assert df_pl.to_dicts() == df_pd.to_dict(
        orient="records"
    ), f"{pformat(df_pl)}\n{pformat(df_pd)}"


@pytest.mark.lite
class TestDataframe:
    """
    Consistency tests across pandas, polars, and hover dataframes.
    """

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_basics(self, df_data):
        """
        Constructor, `()`, `construct`, `copy`, `to_pandas`, `columns`, `shape`.
        """
        pd_df = pd.DataFrame(df_data)
        pl_df = pl.DataFrame(df_data)
        df_pd = PandasDataframe(pd_df)
        df_pl = PolarsDataframe(pl_df)

        assert df_pd() is pd_df
        assert df_pl() is pl_df

        assert df_pd().equals(PandasDataframe.construct(df_data)())
        assert df_pl().frame_equal(PolarsDataframe.construct(df_data)())

        assert df_pd.copy()() is not df_pd()
        assert df_pl.copy()() is not df_pl()

        assert df_pd().equals(df_pl().to_pandas())
        assert df_pd().equals(df_pd.to_pandas())
        assert df_pd().equals(df_pl.to_pandas())

        assert (df_pd.columns == df_pd().columns).all()
        assert df_pl.columns == df_pl().columns

        assert df_pd.shape == df_pd().shape == df_pl.shape == df_pl().shape

    def test_empty_with_columns(self):
        column_to_type = {"a": str, "b": int, "c": bool}

        df_pd = PandasDataframe.empty_with_columns(column_to_type)
        df_pl = PolarsDataframe.empty_with_columns(column_to_type)
        pd_df = pd.DataFrame(columns=column_to_type.keys())
        pl_df = pl.DataFrame(
            schema={
                col: TYPE_TO_POLARS[_type] for col, _type in column_to_type.items()
            },
        )

        assert df_pd().equals(pd_df)
        assert df_pl().frame_equal(pl_df)
        assert df_pd.shape == df_pl.shape == (0, 3)

    @pytest.mark.parametrize("df_data_a", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("df_data_b", DATAFRAME_VALUE_TEST_CASES[::-1])
    def test_concat_rows(self, df_data_a, df_data_b):
        df_pd_a, df_pl_a, pd_df_a, pl_df_a = get_dataframes(df_data_a)
        df_pd_b, df_pl_b, pd_df_b, pl_df_b = get_dataframes(df_data_b)
        df_pd_ab = PandasDataframe.concat_rows([df_pd_a, df_pd_b])
        df_pl_ab = PolarsDataframe.concat_rows([df_pl_a, df_pl_b])

        pd_df_ab = pd.concat([pd_df_a, pd_df_b], axis=0, ignore_index=True)
        # use diagonal for non-overlapping columns
        pl_df_ab = pl.concat([pl_df_a, pl_df_b], how="diagonal")
        assert df_pd_ab().equals(pd_df_ab)
        assert df_pl_ab().frame_equal(pl_df_ab)
        assert df_pl_ab().to_pandas().equals(pd_df_ab)

        try:
            _ = PandasDataframe.concat_rows([pd_df_a, pd_df_b])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass

        try:
            _ = PolarsDataframe.concat_rows([pl_df_a, pl_df_b])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass

    @pytest.mark.parametrize("values", SERIES_VALUE_TEST_CASES)
    def test_series_class_methods(self, values):
        np_values = np.array(values)
        pd_series = pd.Series(values)
        pl_series = pl.Series(values)
        values_pd = PandasDataframe.series_values(pd_series)
        values_pl = PolarsDataframe.series_values(pl_series)
        pd_values = pd_series.values
        pl_values = pl_series.to_numpy()

        assert np.equal(pd_values, np_values).all()
        assert np.equal(pl_values, np_values).all()
        assert np.equal(values_pd, np_values).all()
        assert np.equal(values_pl, np_values).all()

        list_pd = PandasDataframe.series_tolist(pd_series)
        list_pl = PolarsDataframe.series_tolist(pl_series)

        assert list_pd == list_pl == list(values)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_to_dict_of_lists(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)

        df_pd_dict = df_pd.to_dict_of_lists()
        df_pl_dict = df_pl.to_dict_of_lists()
        pd_df_dict = pd_df.to_dict(orient="list")
        pl_df_dict = pl_df.to_dict(as_series=False)

        assert df_pd_dict == df_pl_dict == pd_df_dict == pl_df_dict

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_to_list_of_dicts(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)

        df_pd_dictl = df_pd.to_list_of_dicts()
        df_pl_dictl = df_pl.to_list_of_dicts()
        pd_df_dictl = pd_df.to_dict(orient="records")
        pl_df_dictl = pl_df.to_dicts()

        assert df_pd_dictl == df_pl_dictl == pd_df_dictl == pl_df_dictl

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_get_row_as_dict(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)

        row_pd = df_pd.get_row_as_dict(0)
        row_pl = df_pl.get_row_as_dict(0)
        pd_row = pd_df.iloc[0].to_dict()
        pl_row = pl_df.row(0, named=True)

        assert row_pd == row_pl == pd_row == pl_row

        try:
            _ = df_pd.get_row_as_dict([0, 1])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass

        try:
            _ = df_pl.get_row_as_dict([0, 1])
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_select_rows(self, df_data, indices):
        if indices is None:
            return

        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)

        df_pd_rows = df_pd.select_rows(indices)()
        df_pl_rows = df_pl.select_rows(indices)()

        indices_list = convert_indices_to_list(indices, size=df_pd.shape[0])
        if len(indices_list) == 0:
            pd_df_rows = pd.DataFrame(columns=pd_df.columns)
            pl_df_rows = pl.DataFrame({}, schema=pl_df.schema)
        else:
            pd_df_rows = pd_df.iloc[indices_list]
            pl_df_rows = pl_df[indices_list]
        assert_equivalent_dataframes(
            df_pd_rows,
            df_pl_rows,
            pd_df_rows,
            pl_df_rows,
        )

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_filter_rows_by_operator(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)

        for _op in [operator.eq, operator.ne, operator.gt, operator.lt]:
            _df_pd_slice = df_pd.filter_rows_by_operator("int", _op, 5)()
            _pd_df_slice = pd_df[pd_df["int"].apply(lambda x: _op(x, 5))].reset_index(
                drop=True
            )
            _df_pl_slice = df_pl.filter_rows_by_operator("int", _op, 5)()
            _pl_df_slice = pl_df[np.where(_op(pl_df["int"], 5))[0]]
            assert_equivalent_dataframes(
                _df_pd_slice,
                _df_pl_slice,
                _pd_df_slice,
                _pl_df_slice,
            )

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_unique(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)

        df_pd_unique = df_pd.unique("bool", keep="last")()
        pd_df_unique = pd_df.drop_duplicates("bool", keep="last").reset_index(drop=True)
        df_pl_unique = df_pl.unique("bool", keep="last")()
        pl_df_unique = pl_df.unique("bool", keep="last", maintain_order=True)

        assert_equivalent_dataframes(
            df_pd_unique,
            df_pl_unique,
            pd_df_unique,
            pl_df_unique,
        )

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_set_column_by_constant(self, df_data, indices):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
        indices_list = (
            list(range(df_pd.shape[0]))
            if indices is None
            else convert_indices_to_list(indices, size=df_pd.shape[0])
        )

        col = df_pd.columns[0]
        value = df_pd.get_row_as_dict(0)[col]

        df_pd.set_column_by_constant(col, value, indices)
        pd_df.loc[indices_list, col] = value
        df_pl.set_column_by_constant(col, value, indices)
        for i in indices_list:
            pl_df[i, col] = value

        assert_equivalent_dataframes(df_pd(), df_pl(), pd_df, pl_df)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_set_column_by_array(self, df_data, indices):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
        indices_list = (
            list(range(df_pd.shape[0]))
            if indices is None
            else convert_indices_to_list(indices, size=df_pd.shape[0])
        )
        col = df_pd.columns[0]
        values = df_pd.select_rows(indices)[col].values

        df_pd.set_column_by_array(col, values, indices)
        pd_df.loc[indices_list, col] = values
        df_pl.set_column_by_array(col, values, indices)
        lookup = dict(zip(indices_list, values))
        pl_df = pl_df.update(
            pl.DataFrame({col: [lookup.get(i, None) for i in range(pl_df.shape[0])]})
        )

        assert_equivalent_dataframes(df_pd(), df_pl(), pd_df, pl_df)

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_column_map(self, df_data, indices):
        for col in HASHABLE_COLUMNS:
            if col not in df_data.keys():
                continue
            df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
            mapping = pd_df[col].value_counts().to_dict()
            if col == "str":
                mapping = {
                    _k: "#b0b0b0" for _k in pd_df[col].value_counts().to_dict().keys()
                }
            indices_list = (
                list(range(df_pd.shape[0]))
                if indices is None
                else convert_indices_to_list(indices, size=df_pd.shape[0])
            )

            pd_df_series = pd_df.loc[indices_list, col].map(mapping)
            df_pd_series = df_pd.column_map(
                col, mapping, indices=indices, output="series"
            )
            df_pl_numpy = df_pl.column_map(
                col, mapping, indices=indices, output="numpy"
            )
            df_pl_list = df_pl.column_map(col, mapping, indices=indices, output="list")
            assert df_pd_series.equals(pd_df_series)
            assert np.equal(df_pl_numpy, pd_df_series.values).all()
            assert df_pl_list == pd_df_series.tolist()

            df_pd.column_map(col, mapping, indices=None, as_column="result")
            df_pl.column_map(col, mapping, indices=None, as_column="result")
            assert not df_pd().equals(pd_df)
            assert not df_pl().frame_equal(pl_df)
            assert df_pd().equals(df_pl.to_pandas())

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_column_isin(self, df_data, indices):
        for col in HASHABLE_COLUMNS:
            if col not in df_data.keys():
                continue
            df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
            lookup = set(pd_df.loc[::2, col].values)
            indices_list = (
                list(range(df_pd.shape[0]))
                if indices is None
                else convert_indices_to_list(indices, size=df_pd.shape[0])
            )

            pd_df_series = pd_df.loc[indices_list, col].isin(lookup)
            df_pd_series = df_pd.column_isin(
                col, lookup, indices=indices, output="series"
            )
            df_pl_numpy = df_pl.column_isin(
                col, lookup, indices=indices, output="numpy"
            )
            df_pl_list = df_pl.column_isin(col, lookup, indices=indices, output="list")
            assert df_pd_series.equals(pd_df_series)
            assert np.equal(df_pl_numpy, pd_df_series.values).all()
            assert df_pl_list == pd_df_series.tolist()

            df_pd.column_isin(col, lookup, indices=None, as_column="result")
            df_pl.column_isin(col, lookup, indices=None, as_column="result")
            assert not df_pd().equals(pd_df)
            assert not df_pl().frame_equal(pl_df)
            assert df_pd().equals(df_pl.to_pandas())

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_column_apply(self, df_data, indices):
        df_tmp, _, _, _ = get_dataframes(df_data)

        def func(x):
            if isinstance(x, (str, int, float)):
                return x * 2
            elif hasattr(x, "__iter__"):
                return sum(x)
            else:
                return str(x)

        for col in df_tmp.columns:
            df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
            indices_list = (
                list(range(df_pd.shape[0]))
                if indices is None
                else convert_indices_to_list(indices, size=df_pd.shape[0])
            )

            pd_df_series = pd_df.loc[indices_list, col].apply(func)
            df_pd_series = df_pd.column_apply(
                col, func, indices=indices, output="series"
            )
            df_pl_numpy = df_pl.column_apply(col, func, indices=indices, output="numpy")
            df_pl_list = df_pl.column_apply(col, func, indices=indices, output="list")
            assert df_pd_series.equals(pd_df_series)
            assert np.equal(df_pl_numpy, pd_df_series.values).all()
            assert df_pl_list == pd_df_series.tolist()

            df_pd.column_apply(col, func, indices=None, as_column="result")
            df_pl.column_apply(col, func, indices=None, as_column="result")
            assert not df_pd().equals(pd_df)
            assert not df_pl().frame_equal(pl_df)
            assert df_pd().equals(df_pl.to_pandas())

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    @pytest.mark.parametrize("indices", ROW_INDICES_TEST_CASES)
    def test_row_apply(self, df_data, indices):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
        indices_list = (
            list(range(df_pd.shape[0]))
            if indices is None
            else convert_indices_to_list(indices, size=df_pd.shape[0])
        )

        def func(row):
            return str(row["int"])

        pd_df_series = pd_df.loc[indices_list].apply(func, axis=1)
        df_pd_series = df_pd.row_apply(func, indices=indices, output="series")
        df_pl_numpy = df_pl.row_apply(func, indices=indices, output="numpy")
        df_pl_list = df_pl.row_apply(func, indices=indices, output="list")
        assert df_pd_series.equals(pd_df_series)
        assert np.equal(df_pl_numpy, pd_df_series.values).all()
        assert df_pl_list == pd_df_series.tolist()

        df_pd.row_apply(func, indices=None, as_column="result")
        df_pl.row_apply(func, indices=None, as_column="result")
        assert not df_pd().equals(pd_df)
        assert not df_pl().frame_equal(pl_df)
        assert df_pd().equals(df_pl.to_pandas())

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_get_cell_by_row_column(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
        for col in df_pd.columns:
            row = np.random.randint(0, df_pd.shape[0])

            df_pd_val = df_pd.get_cell_by_row_column(row, col)
            df_pl_val = df_pl.get_cell_by_row_column(row, col)
            pd_df_val = pd_df.at[row, col]
            pl_df_val = pl_df.row(row, named=True)[col]

            if isinstance(df_pd_val, np.ndarray):
                assert np.equal(df_pd_val, df_pl_val).all()
                assert np.equal(df_pd_val, pd_df_val).all()
                assert np.equal(df_pd_val, pl_df_val).all()
            else:
                assert df_pd_val == df_pl_val == pd_df_val == pl_df_val

    @pytest.mark.parametrize("df_data", DATAFRAME_VALUE_TEST_CASES)
    def test_set_cell_by_row_column(self, df_data):
        df_pd, df_pl, pd_df, pl_df = get_dataframes(df_data)
        for col in df_pd.columns:
            row = np.random.randint(0, df_pd.shape[0] // 2)
            old_value = df_pd.get_cell_by_row_column(row, col)
            value = df_pd.get_cell_by_row_column(df_pd.shape[0] - 1 - row, col)
            value = numpy_to_native(value)

            # as of Apr 2023: pyarrow does not support assigning a list to a polars cell
            tolerated = pl.exceptions.ArrowError if isinstance(value, list) else None

            try:
                df_pd.set_cell_by_row_column(row, col, value)
                df_pl.set_cell_by_row_column(row, col, value)
                assert pd_df.at[row, col] == pl_df[row, col] == old_value

                df_pd_val = df_pd.get_cell_by_row_column(row, col)
                df_pl_val = df_pl.get_cell_by_row_column(row, col)
                assert df_pd_val == df_pl_val == value

                pd_df.at[row, col] = value
                pl_df[row, col] = value
                assert_equivalent_dataframes(df_pd(), df_pl(), pd_df, pl_df)
            except Exception as e:
                if tolerated is None or not isinstance(e, tolerated):
                    raise e


@pytest.mark.benchmark
class TestDataframeBenchmark:
    """
    Benchamarking - responsible for performance, not correctness.
    """

    def _benchmark_subroutine(self, pd_, pdx_, plx_, pdms, pdcoeff, plcoeff, runs=10):
        """
        Run callbacks and check that they are faster than the given thresholds.
        Pandas native class callbacks have absolute thresholds.
        Hover wrapped class callbacks have relative (to pandas) thresholds.
        """

        def tic():
            return time.time() * 1000

        pdms_avg, pdx_avg, plx_avg = 0, 0, 0

        gc.disable()
        for _ in range(runs):
            gc.collect()
            pdms_l = tic()
            pd_()
            pdms_r = tic()
            pdms_avg += pdms_r - pdms_l

            pdx_l = tic()
            pdx_()
            pdx_r = tic()
            pdx_avg += pdx_r - pdx_l

            plx_l = tic()
            plx_()
            plx_r = tic()
            plx_avg += plx_r - plx_l

        gc.enable()
        pdms_avg, pdx_avg, plx_avg = pdms_avg / runs, pdx_avg / runs, plx_avg / runs

        errors = []
        if pdms_avg > pdms:
            errors.append(f"Pandas took too long: {pdms_avg} > {pdms}")
        if pdms_avg * pdcoeff < pdx_avg:
            errors.append(
                f"Pandas (wrapped) overhead: {pdx_avg} > {pdms_avg} * {pdcoeff}"
            )
        if pdms_avg * plcoeff < plx_avg:
            errors.append(
                f"Polars (wrapped) overhead: {plx_avg} > {pdms_avg} * {plcoeff}"
            )

        assert not errors, "\n".join(errors)

    @pytest.mark.parametrize(
        "size,types,pdms,pdcoeff,plcoeff",
        [
            (50000, ("int",), 15, 1.05, 2.0),
            (500000, ("int",), 150, 1.01, 2.0),
        ],
    )
    def test_benchmark_concat_rows(self, size, types, pdms, pdcoeff, plcoeff):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        self._benchmark_subroutine(
            lambda: pd.concat([pd_df, pd_df], axis=0, sort=False, ignore_index=True),
            lambda: PandasDataframe.concat_rows([df_pd, df_pd]),
            lambda: PolarsDataframe.concat_rows([df_pl, df_pl]),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int",), 120, 1.01, 2.0),
        ],
    )
    def test_benchmark_select_rows(self, size, types, pdms, pdcoeff, plcoeff):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        maxidx, draw_size = df_pd.shape[0], df_pd.shape[0] // 5
        indices = sorted(set(np.random.randint(0, maxidx, size=draw_size).tolist()))

        self._benchmark_subroutine(
            lambda: pd_df.iloc[indices],
            lambda: df_pd.select_rows(indices),
            lambda: df_pl.select_rows(indices),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int",), 10, 1.01, 2.5),
        ],
    )
    def test_benchmark_filter_rows_by_operator(
        self, size, types, pdms, pdcoeff, plcoeff
    ):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)
        col, val = types[0], pd_df.at[0, types[0]]

        self._benchmark_subroutine(
            lambda: pd_df[pd_df[col] == val],
            lambda: df_pd.filter_rows_by_operator(col, operator.eq, val),
            lambda: df_pl.filter_rows_by_operator(col, operator.eq, val),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int",), 100, 1.01, 3.0),
        ],
    )
    def test_benchmark_set_column_by_constant(
        self, size, types, pdms, pdcoeff, plcoeff
    ):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        maxidx, draw_size = df_pd.shape[0], df_pd.shape[0] // 5
        indices = sorted(set(np.random.randint(0, maxidx, size=draw_size).tolist()))
        col, val = types[0], pd_df.at[0, types[0]]

        def pd_func():
            pd_df.loc[indices, col] = val
            return pd_df

        self._benchmark_subroutine(
            pd_func,
            lambda: df_pd.set_column_by_constant(col, val, indices),
            lambda: df_pl.set_column_by_constant(col, val, indices),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int",), 40, 1.01, 2.0),
            (500000, ("str",), 110, 1.01, 7.0),
        ],
    )
    def test_benchmark_column_map(self, size, types, pdms, pdcoeff, plcoeff):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        col = types[0]
        mapping = {v: v * 2 for v in pd_df[col].unique().tolist()}
        self._benchmark_subroutine(
            lambda: pd_df[col].map(mapping),
            lambda: df_pd.column_map(col, mapping),
            lambda: df_pl.column_map(col, mapping),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int",), 20, 1.01, 2.0),
            (500000, ("str",), 45, 1.01, 3.0),
        ],
    )
    def test_benchmark_column_isin(self, size, types, pdms, pdcoeff, plcoeff):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        col = types[0]
        lookup = set(pd_df[col].loc[::10].tolist())
        self._benchmark_subroutine(
            lambda: pd_df[col].isin(lookup),
            lambda: df_pd.column_isin(col, lookup),
            lambda: df_pl.column_isin(col, lookup),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,func,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int",), lambda x: x**3, 1000, 1.01, 0.8),
            (500000, ("str",), lambda x: len(x), 1000, 1.01, 1.0),
        ],
    )
    def test_benchmark_column_apply(self, size, types, func, pdms, pdcoeff, plcoeff):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        col = types[0]
        self._benchmark_subroutine(
            lambda: pd_df[col].apply(func),
            lambda: df_pd.column_apply(col, func),
            lambda: df_pl.column_apply(col, func),
            pdms,
            pdcoeff,
            plcoeff,
        )

    @pytest.mark.parametrize(
        "size,types,func,pdms,pdcoeff,plcoeff",
        [
            (500000, ("int", "str"), lambda row: row["int"] ** 3, 5000, 1.01, 0.25),
            (500000, ("int", "str"), lambda row: len(row["str"]), 5000, 1.01, 0.25),
        ],
    )
    def test_benchmark_row_apply(self, size, types, func, pdms, pdcoeff, plcoeff):
        df_data = generate_benchmark_dataframe(size, types=types)
        df_pd, df_pl, pd_df, _ = get_dataframes(df_data)

        self._benchmark_subroutine(
            lambda: pd_df.apply(func, axis=1),
            lambda: df_pd.row_apply(func),
            lambda: df_pl.row_apply(func),
            pdms,
            pdcoeff,
            plcoeff,
        )
