import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic.materializers import ArrowMaterializer


def check_for_pyarrow():
    try:
        import pyarrow

        return False
    except ImportError:
        return True


ARROW_TESTS = {
    "a": (["Intercept", "a"], ["Intercept", "a"]),
    "A": (
        ["Intercept", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[T.a]", "A[T.b]", "A[T.c]"],
    ),
    "C(A)": (
        ["Intercept", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[T.a]", "C(A)[T.b]", "C(A)[T.c]"],
    ),
    "a:A": (
        ["Intercept", "A[T.a]:a", "A[T.b]:a", "A[T.c]:a"],
        ["Intercept", "A[T.a]:a", "A[T.b]:a", "A[T.c]:a"],
    ),
}


@pytest.mark.skipif(
    check_for_pyarrow(), reason="PyArrow is required to run the arrow tests."
)
class TestArrowMaterializer:
    @pytest.fixture
    def data(self):
        import pyarrow

        return pyarrow.Table.from_pandas(
            pandas.DataFrame({"a": [1, 2, 3], "A": ["a", "b", "c"]})
        )

    @pytest.fixture
    def materializer(self, data):
        return ArrowMaterializer(data)

    @pytest.mark.parametrize("formula,tests", ARROW_TESTS.items())
    def test_get_model_matrix(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.columns) == tests[0]

        mm = materializer.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.columns) == tests[1]

    @pytest.mark.parametrize("formula,tests", ARROW_TESTS.items())
    def test_get_model_matrix_sparse(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=True, output="sparse"
        )
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.model_spec.feature_names) == tests[0]

        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=False, output="sparse"
        )
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.model_spec.feature_names) == tests[1]

    def test_state(self, materializer):
        import pyarrow

        mm = materializer.get_model_matrix("center(a) - 1")
        assert isinstance(mm, pandas.DataFrame)
        assert list(mm.columns) == ["center(a)"]
        assert numpy.allclose(mm["center(a)"], [-1, 0, 1])

        mm2 = ArrowMaterializer(
            pyarrow.Table.from_pandas(pandas.DataFrame({"a": [4, 5, 6]}))
        ).get_model_matrix(mm.model_spec)
        assert isinstance(mm2, pandas.DataFrame)
        assert list(mm2.columns) == ["center(a)"]
        assert numpy.allclose(mm2["center(a)"], [2, 3, 4])

    def test_missing_field(self, materializer):
        with pytest.raises(KeyError):
            materializer.data_context["invalid_key"]
