import pickle
import re
from io import BytesIO

import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic import ModelMatrices
from formulaic.errors import (
    FactorEncodingError,
    FactorEvaluationError,
    FormulaMaterializationError,
)
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.base import EncodedTermStructure
from formulaic.materializers.types import EvaluatedFactor, FactorValues, NAAction
from formulaic.model_spec import ModelSpec
from formulaic.parser.types import Factor
from formulaic.utils.structured import Structured

PANDAS_TESTS = {
    # '<formula>': (<full_rank_names>, <names>, <full_rank_null_names>, <null_rows>)
    "a": (["Intercept", "a"], ["Intercept", "a"], ["Intercept", "a"], 2),
    "A": (
        ["Intercept", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[a]", "A[b]", "A[c]"],
        ["Intercept", "A[T.c]"],
        2,
    ),
    "C(A)": (
        ["Intercept", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[a]", "C(A)[b]", "C(A)[c]"],
        ["Intercept", "C(A)[T.c]"],
        2,
    ),
    "A:a": (
        ["Intercept", "A[a]:a", "A[b]:a", "A[c]:a"],
        ["Intercept", "A[a]:a", "A[b]:a", "A[c]:a"],
        ["Intercept", "A[a]:a"],
        1,
    ),
    "A:B": (
        [
            "Intercept",
            "B[T.b]",
            "B[T.c]",
            "A[T.b]:B[a]",
            "A[T.c]:B[a]",
            "A[T.b]:B[b]",
            "A[T.c]:B[b]",
            "A[T.b]:B[c]",
            "A[T.c]:B[c]",
        ],
        [
            "Intercept",
            "A[a]:B[a]",
            "A[b]:B[a]",
            "A[c]:B[a]",
            "A[a]:B[b]",
            "A[b]:B[b]",
            "A[c]:B[b]",
            "A[a]:B[c]",
            "A[b]:B[c]",
            "A[c]:B[c]",
        ],
        ["Intercept"],
        1,
    ),
    ".": (
        ["Intercept", "a", "b", "A[T.b]", "A[T.c]", "B[T.b]", "B[T.c]"],
        [
            "Intercept",
            "a",
            "b",
            "A[a]",
            "A[b]",
            "A[c]",
            "B[a]",
            "B[b]",
            "B[c]",
            "D[a]",
        ],
        ["Intercept", "a", "b"],
        1,
    ),
}


class TestPandasMaterializer:
    @pytest.fixture
    def data(self):
        return pandas.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
                "A": ["a", "b", "c"],
                "B": ["a", "b", "c"],
                "D": ["a", "a", "a"],
            }
        )

    @pytest.fixture
    def data_with_nulls(self):
        return pandas.DataFrame(
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
        return PandasMaterializer(data)

    def test_data_conversion(self):
        df = PandasMaterializer({"a": [1, 2, 3]}).data
        assert isinstance(df, pandas.DataFrame)
        assert df.columns == ["a"]

        df2 = PandasMaterializer({"a": 1}).data
        assert isinstance(df2, pandas.DataFrame)
        assert df2.columns == ["a"]
        assert list(df2["a"]) == [1]

        df3 = PandasMaterializer(
            numpy.recarray((2,), dtype=[("x", int), ("y", float), ("z", int)])
        ).data
        assert isinstance(df3, pandas.DataFrame)
        assert list(df3.columns) == ["x", "y", "z"]
        assert len(df3["x"]) == 2

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

    def test_get_model_matrix_edge_cases(self, materializer):
        mm = materializer.get_model_matrix(("a",), ensure_full_rank=True)
        assert isinstance(mm, ModelMatrices)
        assert isinstance(mm[0], pandas.DataFrame)

        mm = materializer.get_model_matrix("a ~ A", ensure_full_rank=True)
        assert isinstance(mm, ModelMatrices)
        assert "lhs" in mm.model_spec
        assert "rhs" in mm.model_spec

        mm = materializer.get_model_matrix(("a ~ A",), ensure_full_rank=True)
        assert isinstance(mm, ModelMatrices)
        assert isinstance(mm[0], ModelMatrices)

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
        assert list(mm.model_spec.column_names) == tests[0]

        mm = materializer.get_model_matrix(
            formula, ensure_full_rank=False, output="sparse"
        )
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.model_spec.column_names) == tests[1]

    def test_get_model_matrix_invalid_output(self, materializer):
        with pytest.raises(
            FormulaMaterializationError,
            match=r"Nominated output .* is invalid\. Available output types are: ",
        ):
            materializer.get_model_matrix(
                "a", ensure_full_rank=True, output="invalid_output"
            )

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
    @pytest.mark.parametrize("output", ["pandas", "numpy", "sparse"])
    def test_na_handling(self, data_with_nulls, formula, tests, output):
        mm = PandasMaterializer(data_with_nulls).get_model_matrix(
            formula, output=output
        )
        if output == "pandas":
            assert isinstance(mm, pandas.DataFrame)
            assert list(mm.columns) == tests[2]
        elif output == "numpy":
            assert isinstance(mm, numpy.ndarray)

        else:
            assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (tests[3], len(tests[2]))
        assert list(mm.model_spec.column_names) == tests[2]

        if formula == "A:B":
            return

        mm = PandasMaterializer(data_with_nulls).get_model_matrix(
            formula, na_action="ignore"
        )
        assert isinstance(mm, pandas.DataFrame)
        if formula == ".":
            assert mm.shape == (3, 5)
        else:
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
            ModelSpec(formula=[]),
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
                ModelSpec(formula=[]),
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
                ModelSpec(formula=[], encoder_state={"a": ("categorical", {})}),
                drop_rows=set(),
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
                spec=ModelSpec(formula=[]),
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
            spec=ModelSpec(formula=[]),
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
            spec=ModelSpec(formula=[]),
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
                spec=ModelSpec(formula=[]),
                drop_rows=[],
            )["A[a]"]
        ) == [1, 2, 3]

        assert list(
            materializer._encode_evaled_factor(
                factor=EvaluatedFactor(
                    factor=Factor("B", eval_method="python", kind="categorical"),
                    values=FactorValues({"a": ["a", "b", "c"]}, kind="categorical"),
                ),
                spec=ModelSpec(formula=[]),
                drop_rows=[],
            )
        ) == ["B[a][a]", "B[a][b]", "B[a][c]"]

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
        assert list(m.columns) == ["A[a]", "A[b]", "A[c]"]
        assert list(m.model_spec.get_model_matrix(data3).columns) == [
            "A[a]",
            "A[b]",
            "A[c]",
        ]

        m2 = PandasMaterializer(data2).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m2.columns) == ["A[a]", "A[b]", "A[c]"]
        assert list(m2.model_spec.get_model_matrix(data3).columns) == [
            "A[a]",
            "A[b]",
            "A[c]",
        ]

        m3 = PandasMaterializer(data3).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m3.columns) == ["A[c]", "A[b]", "A[a]"]
        assert list(m3.model_spec.get_model_matrix(data).columns) == [
            "A[c]",
            "A[b]",
            "A[a]",
        ]

    def test_term_clustering(self, materializer):
        assert materializer.get_model_matrix(
            "a + b + a:A + b:A"
        ).model_spec.column_names == (
            "Intercept",
            "a",
            "b",
            "a:A[T.b]",
            "a:A[T.c]",
            "b:A[T.b]",
            "b:A[T.c]",
        )
        assert materializer.get_model_matrix(
            "a + b + a:A + b:A", cluster_by="numerical_factors"
        ).model_spec.column_names == (
            "Intercept",
            "a",
            "a:A[T.b]",
            "a:A[T.c]",
            "b",
            "b:A[T.b]",
            "b:A[T.c]",
        )

    def test_model_spec_pickleable(self, materializer):
        o = BytesIO()
        ms = materializer.get_model_matrix("a ~ a:A")
        pickle.dump(ms.model_spec, o)
        o.seek(0)
        ms2 = pickle.load(o)
        assert isinstance(ms, Structured)
        assert ms2.lhs.formula == ["a"]

    def test_no_levels_encoding(self, data):
        mm = PandasMaterializer(data, output="pandas").get_model_matrix("a + D")

        assert mm.model_spec.column_names == ("Intercept", "a")
        assert mm.shape == (3, 2)

        mm = PandasMaterializer(data, output="sparse").get_model_matrix("a + D")

        assert mm.model_spec.column_names == ("Intercept", "a")
        assert mm.shape == (3, 2)

        mm = PandasMaterializer(data, output="numpy").get_model_matrix("a + D")

        assert mm.model_spec.column_names == ("Intercept", "a")
        assert mm.shape == (3, 2)

    def test_none_values(self, data):
        mm = PandasMaterializer(data, output="pandas").get_model_matrix("{None} -1")
        assert mm.shape == (3, 0)
        assert len(mm.model_spec.structure) == 1
        assert mm.model_spec.structure == [
            EncodedTermStructure(term="None", scoped_terms=[], columns=[]),
        ]

    def test_quoted_python_args(self):
        data = pandas.DataFrame({"exotic!~  -name": [1, 2, 3]})
        mm = PandasMaterializer(data, output="pandas").get_model_matrix(
            "np.power(`exotic!~  -name`, 2)"
        )
        assert mm.shape == (3, 2)
        assert len(mm.model_spec.structure) == 2
        assert numpy.all(mm.values == numpy.array([[1, 1], [1, 4], [1, 9]]))

    def test_lookup_nonexistent_variable(self):
        data = pandas.DataFrame({})
        with pytest.raises(
            FactorEvaluationError,
            match=re.escape(
                "Unable to evaluate factor `a`. [NameError: `a` is not present in the dataset or evaluation context.]"
            ),
        ):
            PandasMaterializer(data).get_model_matrix("a")
        with pytest.raises(
            FactorEvaluationError,
            match=re.escape(
                "Unable to evaluate factor `I(a)`. [NameError: name 'a' is not defined]"
            ),
        ):
            PandasMaterializer(data).get_model_matrix("I(a)")

    def test_transform_state_with_inconsistent_formatting(self, data):
        ms1 = PandasMaterializer(data).get_model_matrix("bs(a, df=4)").model_spec
        ms2 = PandasMaterializer(data).get_model_matrix("bs( `a`, df = 4) ").model_spec
        assert ms1.transform_state == ms2.transform_state

    def test_nested_transform_state(self, data):
        ms = PandasMaterializer(data).get_model_matrix("bs(bs(a))").model_spec
        assert {"bs(a)", "bs(bs(a))"}.issubset(ms.transform_state)

    def test_drop_rows(self, data, data_with_nulls):
        drop_rows = {0, 1}
        mm = PandasMaterializer(data).get_model_matrix("a", drop_rows=drop_rows)
        assert mm.shape == (1, 2)
        assert list(mm.index) == [2]
        assert drop_rows == {0, 1}

        drop_rows = {0, 1}
        mm = PandasMaterializer(data_with_nulls).get_model_matrix(
            "a", drop_rows=drop_rows
        )
        assert mm.shape == (0, 2)
        assert not list(mm.index)
        assert drop_rows == {0, 1, 2}
