import re

import narwhals.stable.v1 as nw
import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic import model_matrix
from formulaic.materializers import NarwhalsMaterializer

NARWHALS_TESTS = {
    "a": (["Intercept", "a"], ["Intercept", "a"]),
    "A": (
        ["Intercept", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[a]", "A[b]", "A[c]"],
    ),
    "C(A)": (
        ["Intercept", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[a]", "C(A)[b]", "C(A)[c]"],
    ),
    "a:A": (
        ["Intercept", "a:A[a]", "a:A[b]", "a:A[c]"],
        ["Intercept", "a:A[a]", "a:A[b]", "a:A[c]"],
    ),
}


class TestNarwhalsMaterializer:
    @pytest.fixture(params=["pandas", "arrow", "polars"])
    def data(self, request):
        if request.param == "pandas":
            df = pandas.DataFrame({"a": [1, 2, 3], "A": ["a", "b", "c"]})
            if request.param == "arrow":
                pyarrow = pytest.importorskip("pyarrow")
                return pyarrow.Table.from_pandas(df)
            return df

        polars = pytest.importorskip("polars")
        return polars.DataFrame({"a": [1, 2, 3], "A": ["a", "b", "c"]})

    @pytest.fixture
    def data_with_nulls(self):
        polars = pytest.importorskip("polars")
        return polars.DataFrame(
            {
                "a": [1, 2, None],
                "b": [1, 2, 3],
                "A": ["a", None, "c"],
                "B": ["a", "b", None],
                "D": ["a", "a", "a"],
            }
        )

    @pytest.fixture
    def materializer(self, data):
        return NarwhalsMaterializer(data)

    def test_data_wrapper(self, materializer):
        assert set(materializer.data_context) == {"a", "A"}
        assert len(materializer.data_context) == 2

    @pytest.mark.parametrize("formula,tests", NARWHALS_TESTS.items())
    @pytest.mark.parametrize("output", ["narwhals", "pandas", "numpy"])
    def test_get_model_matrix(self, materializer, formula, tests, data, output):
        if output == "pandas":
            target_type = pandas.DataFrame
            get_cols = lambda mm: mm.columns
        elif output == "numpy":
            target_type = numpy.ndarray
            get_cols = lambda mm: mm.model_spec.column_names
        else:
            target_type = type(data)
            get_cols = (
                lambda mm: mm.column_names if "pyarrow" in str(type(mm)) else mm.columns
            )

        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=True, output=output
        )
        assert isinstance(mm, target_type)
        assert mm.shape == (3, len(tests[0]))
        assert list(get_cols(mm)) == tests[0]

        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=False, output=output
        )
        assert isinstance(mm, target_type)
        assert mm.shape == (3, len(tests[1]))
        assert list(get_cols(mm)) == tests[1]

    @pytest.mark.parametrize("formula,tests", NARWHALS_TESTS.items())
    def test_get_model_matrix_sparse(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=True, output="sparse"
        )
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.model_spec.column_names) == tests[0]

        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=False, output="sparse"
        )
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.model_spec.column_names) == tests[1]

    def test_missing_field(self, materializer):
        with pytest.raises(KeyError):
            materializer.data_context["invalid_key"]

    @pytest.mark.parametrize("formula,tests", NARWHALS_TESTS.items())
    @pytest.mark.parametrize("output", ["pandas", "narwhals", "numpy"])
    def test_na_handling(self, data_with_nulls, formula, tests, output):
        polars = pytest.importorskip("polars")
        mm = NarwhalsMaterializer(data_with_nulls).get_model_matrix(
            formula, output=output
        )

        mm = NarwhalsMaterializer(data_with_nulls).get_model_matrix(
            formula, na_action="ignore"
        )
        assert isinstance(mm, polars.DataFrame)
        if formula == ".":
            assert mm.shape == (3, 5)
        else:
            assert mm.shape == (3, len(tests[0]) + (-1 if "A" in formula else 0))

        if formula != "C(A)":  # C(A) pre-encodes the data, stripping out nulls.
            with pytest.raises(ValueError):
                NarwhalsMaterializer(data_with_nulls).get_model_matrix(
                    formula, na_action="raise"
                )

    def test_data_registration(self):
        polars = pytest.importorskip("polars")
        df = polars.DataFrame(
            {
                "xm": [1.0, 4.0, None, 2.0],
            }
        )
        result = model_matrix("xm", data=nw.from_native(df), na_action="drop")
        assert result.to_dict(as_series=False) == {
            "Intercept": [1.0, 1.0, 1.0],
            "xm": [1.0, 4.0, 2.0],
        }

    def test_empty(self, materializer):
        mm = materializer.get_model_matrix("0", ensure_full_rank=True)
        assert mm.shape[1] == 0
        mm = materializer.get_model_matrix("0", ensure_full_rank=False)
        assert mm.shape[1] == 0

        mm = materializer.get_model_matrix("0", output="numpy")
        assert mm.shape[1] == 0

        mm = materializer.get_model_matrix("0", output="sparse")
        assert mm.shape[1] == 0

        mm = materializer.get_model_matrix("0", output="pandas")
        assert mm.shape[1] == 0

    def test_narwhals_frame(self):
        NarwhalsMaterializer(
            data=nw.from_native(pandas.DataFrame({"a": [1, 2, 3]}))
        ).get_model_matrix("0")
