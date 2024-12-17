import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic.materializers import NarwhalsMaterializer

pyarrow = pytest.importorskip("pyarrow")


ARROW_TESTS = {
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


class TestNarwhalsMaterializerArrow:
    @pytest.fixture
    def data(self):
        return pyarrow.Table.from_pandas(
            pandas.DataFrame({"a": [1, 2, 3], "A": ["a", "b", "c"]})
        )

    @pytest.fixture
    def materializer(self, data):
        return NarwhalsMaterializer(data)

    def test_data_wrapper(self, materializer):
        assert set(materializer.data_context) == {"a", "A"}
        assert len(materializer.data_context) == 2

    @pytest.mark.parametrize("formula,tests", ARROW_TESTS.items())
    def test_get_model_matrix(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, pyarrow.Table)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.column_names) == tests[0]

        mm = materializer.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, pyarrow.Table)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.column_names) == tests[1]

    @pytest.mark.parametrize("formula,tests", ARROW_TESTS.items())
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
        assert isinstance(mm, pyarrow.Table)
        assert list(mm.column_names) == ["center(a)"]
        assert numpy.allclose(mm["center(a)"], [-1, 0, 1])

        mm2 = NarwhalsMaterializer(
            pyarrow.Table.from_pandas(pandas.DataFrame({"a": [4, 5, 6]}))
        ).get_model_matrix(mm.model_spec)
        assert isinstance(mm2, pyarrow.Table)
        assert list(mm2.column_names) == ["center(a)"]
        assert numpy.allclose(mm2["center(a)"], [2, 3, 4])

    def test_missing_field(self, materializer):
        with pytest.raises(KeyError):
            materializer.data_context["invalid_key"]
