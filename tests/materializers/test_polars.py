import narwhals.stable.v1 as nw
import numpy
import pytest
import scipy.sparse as spsparse

from formulaic import model_matrix
from formulaic.materializers import NarwhalsMaterializer

polars = pytest.importorskip("polars")

POLARS_TESTS = {
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


class TestNarwhalsMaterializerPolars:
    @pytest.fixture
    def data(self):
        import polars

        return polars.DataFrame({"a": [1, 2, 3], "A": ["a", "b", "c"]})

    @pytest.fixture
    def data_with_nulls(self):
        return polars.DataFrame(
            {
                "a": [1, 2, None],
                "b": [1, 2, 3],
                "A": ["a", None, "c"],
                "B": ["a", "b", None],
                "D": ["a", "a", "a"],
            }
        )

    @pytest.mark.parametrize("formula,tests", POLARS_TESTS.items())
    @pytest.mark.parametrize("output", ["pandas", "narwhals", "numpy"])
    def test_na_handling(self, data_with_nulls, formula, tests, output):
        mm = NarwhalsMaterializer(data_with_nulls).get_model_matrix(
            formula, output=output
        )

        if formula == "A:B":
            return

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

    @pytest.fixture
    def materializer(self, data):
        return NarwhalsMaterializer(data)

    def test_data_wrapper(self, materializer):
        assert set(materializer.data_context) == {"a", "A"}
        assert len(materializer.data_context) == 2

    @pytest.mark.parametrize("formula,tests", POLARS_TESTS.items())
    def test_get_model_matrix(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, polars.DataFrame)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.columns) == tests[0]

        mm = materializer.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, polars.DataFrame)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.columns) == tests[1]

    @pytest.mark.parametrize("formula,tests", POLARS_TESTS.items())
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

    def test_state(self, materializer):
        mm = materializer.get_model_matrix("center(a) - 1")
        assert isinstance(mm, polars.DataFrame)
        assert list(mm.columns) == ["center(a)"]
        assert numpy.allclose(mm["center(a)"], [-1, 0, 1])

        mm2 = NarwhalsMaterializer(polars.DataFrame({"a": [4, 5, 6]})).get_model_matrix(
            mm.model_spec
        )
        assert isinstance(mm2, polars.DataFrame)
        assert list(mm2.columns) == ["center(a)"]
        assert numpy.allclose(mm2["center(a)"], [2, 3, 4])

    def test_missing_field(self, materializer):
        with pytest.raises(KeyError):
            materializer.data_context["invalid_key"]

    def test_for_data(self):
        df = polars.DataFrame(
            {
                "xm": [1.0, 4.0, None, 2.0],
            }
        )
        result = model_matrix("xm", data=df, na_action="drop")
        assert result.to_dict(as_series=False) == {
            "Intercept": [1.0, 1.0, 1.0],
            "xm": [1.0, 4.0, 2.0],
        }

    def test_narwhals_input(self):
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
