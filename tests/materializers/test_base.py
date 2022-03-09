import pandas
import pytest

from formulaic.errors import FactorEncodingError, FormulaMaterializerNotFoundError
from formulaic.materializers.types import (
    EvaluatedFactor,
    FactorValues,
    ScopedFactor,
    ScopedTerm,
)
from formulaic.materializers.base import FormulaMaterializer
from formulaic.materializers.pandas import PandasMaterializer
from formulaic.model_spec import ModelSpec
from formulaic.parser.types import Factor


class TestFormulaMaterializer:
    def test_registrations(self):
        assert sorted(FormulaMaterializer.REGISTERED_NAMES) == ["arrow", "pandas"]
        assert sorted(FormulaMaterializer.REGISTERED_INPUTS) == [
            "pandas.core.frame.DataFrame",
            "pyarrow.lib.Table",
        ]

    def test_retrieval(self):
        assert FormulaMaterializer.for_materializer("pandas") is PandasMaterializer
        assert (
            FormulaMaterializer.for_materializer(PandasMaterializer)
            is PandasMaterializer
        )
        assert (
            FormulaMaterializer.for_data(pandas.DataFrame(), output="numpy")
            is PandasMaterializer
        )

        with pytest.raises(FormulaMaterializerNotFoundError):
            FormulaMaterializer.for_materializer("invalid_materializer")

        with pytest.raises(
            FormulaMaterializerNotFoundError,
            match=r"No materializer has been registered for input type",
        ):
            FormulaMaterializer.for_data("str")

        with pytest.raises(
            FormulaMaterializerNotFoundError,
            match=r"No materializer has been registered for input type .* that supports output type",
        ):
            FormulaMaterializer.for_data(pandas.DataFrame(), output="invalid_output")

    def test_defaults(self):
        assert (
            FormulaMaterializer.for_data(pandas.DataFrame({"a": [1, 2, 3]}))
            is PandasMaterializer
        )

    @pytest.fixture
    def evaled_factors(self):
        return [
            EvaluatedFactor(
                factor=Factor("A"),
                values=FactorValues(
                    pandas.Series([1, 2, 3, 4]),
                    kind="categorical",
                    spans_intercept=True,
                ),
            ),
            EvaluatedFactor(
                factor=Factor("b"),
                values=FactorValues(
                    pandas.Series([1, 2, 3, 4]),
                    kind="numerical",
                    spans_intercept=False,
                ),
            ),
        ]

    def test__get_scoped_terms_spanned_by_evaled_factors(self, evaled_factors):
        assert sorted(
            [
                str(st)
                for st in FormulaMaterializer._get_scoped_terms_spanned_by_evaled_factors(
                    evaled_factors
                )
            ]
        ) == sorted(["A-:b", "b"])

    def test__simplify_scoped_terms(self, evaled_factors):
        A, B, C = [ScopedFactor(l, reduced=False) for l in "ABC"]
        A_, B_, C_ = [ScopedFactor(l, reduced=True) for l in "ABC"]
        assert FormulaMaterializer._simplify_scoped_terms(
            [
                ScopedTerm((C_,)),
                ScopedTerm((A_, C_)),
                ScopedTerm((B_, C_)),
                ScopedTerm((A_, B_, C_)),
            ]
        ) == [ScopedTerm((A, B, C_))]

    def test__flatten_encoded_evaled_factor(self):

        flattened = PandasMaterializer(data=None)._flatten_encoded_evaled_factor(
            "name",
            FactorValues(
                {
                    "a": FactorValues({"1": 1, "2": 2}, format="{name}@{field}"),
                    "b": {"3": 3, "4": 4},
                }
            ),
        )

        assert list(flattened) == ["name[a]@1", "name[a]@2", "name[b][3]", "name[b][4]"]
        assert list(flattened.values()) == [1, 2, 3, 4]

    def test__enforce_structure(self):

        # TODO: Make sure that imputations are intuitive

        df = pandas.DataFrame({"a": [1]})
        cols = [("A", {"A"}, {"a": 1})]

        assert (
            len(
                list(
                    PandasMaterializer(data=None)._enforce_structure(
                        cols=[("A", {"A"}, {"a": 1})],
                        spec=ModelSpec([], structure=[("A", {"A"}, ["a"])]),
                        drop_rows=[],
                    )
                )
            )
            == 1
        )

        # Ensure than an exception is raised if input structure > expected structure
        with pytest.raises(FactorEncodingError):
            list(
                PandasMaterializer(df)._enforce_structure(
                    cols=[("A", {"A"}, {"a": 1, "b": 2})],
                    spec=ModelSpec([], structure=[("A", {"A"}, ["a"])]),
                    drop_rows=[],
                )
            )

        # Ensure that missing columns are imputed
        assert list(
            list(
                PandasMaterializer(df)._enforce_structure(
                    cols=[("A", {"A"}, {"a": 1})],
                    spec=ModelSpec([], structure=[("A", {"A"}, ["a", "b"])]),
                    drop_rows=[],
                )
            )[0][-1]
        ) == ["a", "b"]

        assert list(
            list(
                PandasMaterializer(df)._enforce_structure(
                    cols=[("A", {"A"}, {})],
                    spec=ModelSpec([], structure=[("A", {"A"}, ["a", "b"])]),
                    drop_rows=[],
                )
            )[0][-1]
        ) == ["a", "b"]

        # Ensure that imputation does not occur if it would be ambiguous
        with pytest.raises(FactorEncodingError):
            list(
                PandasMaterializer(df)._enforce_structure(
                    cols=[("A", {"A"}, {"a": 1, "b": 2})],
                    spec=ModelSpec([], structure=[("A", {"A"}, ["a", "b", "c"])]),
                    drop_rows=[],
                )
            )

        # Ensure that an exception is raised if columns do not match
        with pytest.raises(FactorEncodingError):
            list(
                PandasMaterializer(df)._enforce_structure(
                    cols=[("A", {"A"}, {"a": 1, "b": 2, "d": 3})],
                    spec=ModelSpec([], structure=[("A", {"A"}, ["a", "b", "c"])]),
                    drop_rows=[],
                )
            )

    def test__get_columns_for_term(self):
        assert FormulaMaterializer._get_columns_for_term(
            None, [{"a": 1}, {"b": 2}], ModelSpec([]), scale=3
        ) == {"a:b": 6}
