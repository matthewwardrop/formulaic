import pandas
import pytest

from formulaic.errors import FormulaMaterializerNotFoundError
from formulaic.materializers.types import EvaluatedFactor, ScopedFactor, ScopedTerm
from formulaic.materializers.base import FormulaMaterializer
from formulaic.materializers.pandas import PandasMaterializer
from formulaic.parser.types import Factor


class TestFormulaMaterializer:

    def test_registrations(self):
        assert sorted(FormulaMaterializer.REGISTRY) == ['arrow', 'pandas']
        assert sorted(FormulaMaterializer.DEFAULTS) == ['pandas.core.frame.DataFrame', 'pyarrow.lib.Table']

    def test_retrieval(self):
        assert FormulaMaterializer.for_materializer('pandas') is PandasMaterializer
        assert FormulaMaterializer.for_materializer(PandasMaterializer) is PandasMaterializer

        with pytest.raises(FormulaMaterializerNotFoundError):
            FormulaMaterializer.for_materializer('invalid_materializer')

    def test_defaults(self):
        assert FormulaMaterializer.for_data(pandas.DataFrame({'a': [1, 2, 3]})) is PandasMaterializer

    @pytest.fixture
    def evaled_factors(self):
        return [
            EvaluatedFactor(Factor('A'), pandas.Series([1, 2, 3, 4]), kind='categorical', spans_intercept=True),
            EvaluatedFactor(Factor('b'), pandas.Series([1, 2, 3, 4]), kind='numerical', spans_intercept=False),
        ]

    def test__get_scoped_terms_spanned_by_evaled_factors(self, evaled_factors):
        assert sorted([str(st) for st in FormulaMaterializer._get_scoped_terms_spanned_by_evaled_factors(evaled_factors)]) == sorted(['A-:b', 'b'])

    def test__simplify_scoped_terms(self, evaled_factors):
        A, B, C = [
            ScopedFactor(l, reduced=False)
            for l in 'ABC'
        ]
        A_, B_, C_ = [
            ScopedFactor(l, reduced=True)
            for l in 'ABC'
        ]
        assert (
            FormulaMaterializer._simplify_scoped_terms([
                ScopedTerm((C_, )),
                ScopedTerm((A_, C_)),
                ScopedTerm((B_, C_)),
                ScopedTerm((A_, B_, C_)),
            ]) == [ScopedTerm((A, B, C_))]
        )
