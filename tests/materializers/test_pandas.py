import pickle
import re
from io import BytesIO

import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic.errors import (
    FactorEncodingError,
    FactorEvaluationError,
    FormulaMaterializationError,
)
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.types import EvaluatedFactor, FactorValues, NAAction
from formulaic.model_spec import ModelSpec
from formulaic.parser.types import Factor, Structured


PANDAS_TESTS = {
    # '<formula>': (<full_rank_names>, <names>, <full_rank_null_names>, <null_rows>)
    "a": (["Intercept", "a"], ["Intercept", "a"], ["Intercept", "a"], 2),
    "A": (
        ["Intercept", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[T.a]", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[T.c]"],
        2,
    ),
    "C(A)": (
        ["Intercept", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[T.a]", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[T.c]"],
        3,
    ),
    "a:A": (
        ["Intercept", "A[T.a]:a", "A[T.b]:a", "A[T.c]:a"],
        ["Intercept", "A[T.a]:a", "A[T.b]:a", "A[T.c]:a"],
        ["Intercept", "A[T.a]:a"],
        1,
    ),
}


class TestPandasMaterializer:
    @pytest.fixture
    def data(self):
        return pandas.DataFrame({"a": [1, 2, 3], "A": ["a", "b", "c"]})

    @pytest.fixture
    def data_with_nulls(self):
        return pandas.DataFrame({"a": [1, 2, None], "A": ["a", None, "c"]})

    @pytest.fixture
    def materializer(self, data):
        return PandasMaterializer(data)

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
    def test_get_model_matrix(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.columns) == tests[0]

        mm = materializer.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.columns) == tests[1]

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
    def test_get_model_matrix_numpy(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=True, output="numpy"
        )
        assert isinstance(mm, numpy.ndarray)
        assert mm.shape == (3, len(tests[0]))

        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=False, output="numpy"
        )
        assert isinstance(mm, numpy.ndarray)
        assert mm.shape == (3, len(tests[1]))

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
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

    def test_get_model_matrix_invalid_output(self, materializer):
        with pytest.raises(
            FormulaMaterializationError,
            match=r"Nominated output .* is invalid\. Available output types are: ",
        ):
            materializer.get_model_matrix(
                "a", ensure_full_rank=True, output="invalid_output"
            )

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
    def test_na_handling(self, data_with_nulls, formula, tests):
        mm = PandasMaterializer(data_with_nulls).get_model_matrix(formula)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (tests[3], len(tests[2]))
        assert list(mm.columns) == tests[2]

        mm = PandasMaterializer(data_with_nulls).get_model_matrix(
            formula, na_action="ignore"
        )
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[0]) + (-1 if "A" in formula else 0))

        if formula != "C(A)":  # C(A) pre-encodes the data, stripping out nulls.
            with pytest.raises(ValueError):
                PandasMaterializer(data_with_nulls).get_model_matrix(
                    formula, na_action="raise"
                )

    def test_state(self, materializer):
        mm = materializer.get_model_matrix("center(a) - 1")
        assert isinstance(mm, pandas.DataFrame)
        assert list(mm.columns) == ["center(a)"]
        assert numpy.allclose(mm["center(a)"], [-1, 0, 1])

        mm2 = PandasMaterializer(pandas.DataFrame({"a": [4, 5, 6]})).get_model_matrix(
            mm.model_spec
        )
        assert isinstance(mm2, pandas.DataFrame)
        assert list(mm2.columns) == ["center(a)"]
        assert numpy.allclose(mm2["center(a)"], [2, 3, 4])

        mm3 = mm.model_spec.get_model_matrix(pandas.DataFrame({"a": [4, 5, 6]}))
        assert isinstance(mm3, pandas.DataFrame)
        assert list(mm3.columns) == ["center(a)"]
        assert numpy.allclose(mm3["center(a)"], [2, 3, 4])

    def test_factor_evaluation_edge_cases(self, materializer):
        # Test that categorical kinds are set if type would otherwise be numerical
        ev_factor = materializer._evaluate_factor(
            Factor("a", eval_method="lookup", kind="categorical"),
            ModelSpec([]),
            drop_rows=set(),
        )
        assert ev_factor.metadata.kind.value == "categorical"

        # Test that other kind mismatches result in an exception
        materializer.factor_cache = {}
        with pytest.raises(
            FactorEncodingError,
            match=re.escape(
                "Factor `A` is expecting values of kind 'numerical', but they are actually of kind 'categorical'."
            ),
        ):
            materializer._evaluate_factor(
                Factor("A", eval_method="lookup", kind="numerical"),
                ModelSpec([]),
                drop_rows=set(),
            )

        # Test that if an encoding has already been determined, that an exception is raised
        # if the new encoding does not match
        materializer.factor_cache = {}
        with pytest.raises(
            FactorEncodingError,
            match=re.escape(
                "The model specification expects factor `a` to have values of kind `categorical`, but they are actually of kind `numerical`."
            ),
        ):
            materializer._evaluate_factor(
                Factor("a", eval_method="lookup", kind="numerical"),
                ModelSpec([], encoder_state={"a": ("categorical", {})}),
                drop_rows=set(),
            )

        # Test that invalid (kind == UNKNOWN) factors raise errors
        materializer.factor_cache = {}
        with pytest.raises(
            FactorEvaluationError,
            match=re.escape(
                "The evaluation method `unknown` for factor `a` is not understood."
            ),
        ):
            assert materializer._evaluate_factor(
                Factor("a"), ModelSpec([]), drop_rows=set()
            )

    def test__is_categorical(self, materializer):
        assert materializer._is_categorical([1, 2, 3]) is False
        assert materializer._is_categorical(pandas.Series(["a", "b", "c"])) is True
        assert materializer._is_categorical(pandas.Categorical(["a", "b", "c"])) is True
        assert materializer._is_categorical(FactorValues({}, kind="categorical"))

    def test_encoding_edge_cases(self, materializer):
        # Verify that constant encoding works well
        assert list(
            materializer._encode_evaled_factor(
                factor=EvaluatedFactor(
                    factor=Factor("10", eval_method="literal", kind="constant"),
                    values=FactorValues(10, kind="constant"),
                ),
                spec=ModelSpec([]),
                drop_rows=[],
            )["10"]
        ) == [10, 10, 10]

        # Verify that unencoded dictionaries with drop-fields work
        assert materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("a", eval_method="lookup", kind="numerical"),
                values=FactorValues(
                    {"a": [1, 2, 3], "b": [4, 5, 6]},
                    kind="numerical",
                    spans_intercept=True,
                    drop_field="a",
                ),
            ),
            spec=ModelSpec([]),
            drop_rows=set(),
        ) == {
            "a[a]": [1, 2, 3],
            "a[b]": [4, 5, 6],
        }

        assert materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("a", eval_method="lookup", kind="numerical"),
                values=FactorValues(
                    {"a": [1, 2, 3], "b": [4, 5, 6]},
                    kind="numerical",
                    spans_intercept=True,
                    drop_field="a",
                ),
            ),
            spec=ModelSpec([]),
            drop_rows=set(),
            reduced_rank=True,
        ) == {
            "a[b]": [4, 5, 6],
        }

        # Verify that encoding of nested dictionaries works well
        assert list(
            materializer._encode_evaled_factor(
                factor=EvaluatedFactor(
                    factor=Factor("A", eval_method="python", kind="numerical"),
                    values=FactorValues(
                        {"a": [1, 2, 3], "b": [4, 5, 6], "__metadata__": None},
                        kind="numerical",
                    ),
                ),
                spec=ModelSpec([]),
                drop_rows=[],
            )["A[a]"]
        ) == [1, 2, 3]

        assert list(
            materializer._encode_evaled_factor(
                factor=EvaluatedFactor(
                    factor=Factor("B", eval_method="python", kind="categorical"),
                    values=FactorValues({"a": ["a", "b", "c"]}, kind="categorical"),
                ),
                spec=ModelSpec([]),
                drop_rows=[],
            )
        ) == ["B[a][T.a]", "B[a][T.b]", "B[a][T.c]"]

    def test_empty(self, materializer):
        mm = materializer.get_model_matrix("0", ensure_full_rank=True)
        assert mm.shape[1] == 0
        mm = materializer.get_model_matrix("0", ensure_full_rank=False)
        assert mm.shape[1] == 0

        mm = materializer.get_model_matrix("0", output="numpy")
        assert mm.shape[1] == 0

        mm = materializer.get_model_matrix("0", output="sparse")
        assert mm.shape[1] == 0

    def test_index_maintained(self):
        data = pandas.DataFrame(
            {"a": [1, 2, 3], "A": ["a", "b", "c"]}, index=["a", "b", "c"]
        )
        mm = PandasMaterializer(data).get_model_matrix("a + A")
        assert all(mm.index == data.index)

        data = pandas.DataFrame(
            {"a": [None, 2, 3], "A": ["a", None, "c"]}, index=["a", "b", "c"]
        )
        mm = PandasMaterializer(data).get_model_matrix("a + A")
        assert list(mm.index) == ["c"]

        data = pandas.DataFrame(
            {"a": [0, 1, 2, None, 4, 5], "A": list("ABCDEF")}, index=[0, 2, 4, 6, 8, 10]
        )
        mm = PandasMaterializer(data).get_model_matrix("1 + a + A")
        assert list(mm.index) == [0, 2, 4, 8, 10]
        assert not numpy.any(pandas.isnull(mm))

    def test_category_reordering(self):
        data = pandas.DataFrame({"A": ["a", "b", "c"]})
        data2 = pandas.DataFrame({"A": ["c", "b", "a"]})
        data3 = pandas.DataFrame(
            {"A": pandas.Categorical(["c", "b", "a"], categories=["c", "b", "a"])}
        )

        m = PandasMaterializer(data).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m.columns) == ["A[T.a]", "A[T.b]", "A[T.c]"]
        assert list(m.model_spec.get_model_matrix(data3).columns) == [
            "A[T.a]",
            "A[T.b]",
            "A[T.c]",
        ]

        m2 = PandasMaterializer(data2).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m2.columns) == ["A[T.a]", "A[T.b]", "A[T.c]"]
        assert list(m2.model_spec.get_model_matrix(data3).columns) == [
            "A[T.a]",
            "A[T.b]",
            "A[T.c]",
        ]

        m3 = PandasMaterializer(data3).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m3.columns) == ["A[T.c]", "A[T.b]", "A[T.a]"]
        assert list(m3.model_spec.get_model_matrix(data).columns) == [
            "A[T.c]",
            "A[T.b]",
            "A[T.a]",
        ]

    def test_model_spec_pickleable(self, materializer):
        o = BytesIO()
        ms = materializer.get_model_matrix("a ~ a:A")
        pickle.dump(ms.model_spec, o)
        o.seek(0)
        ms2 = pickle.load(o)
        assert isinstance(ms, Structured)
        assert ms2.lhs.formula.terms == ["a"]
