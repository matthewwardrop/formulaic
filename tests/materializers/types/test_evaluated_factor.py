import pytest

from formulaic.parser.types import Factor
from formulaic.materializers.types import EvaluatedFactor


class TestEvaluatedFactor:

    @pytest.fixture
    def ev_factor(self):
        return EvaluatedFactor(Factor('a'), [1, 2, 3], kind='numerical')

    def test_attributes(self, ev_factor):
        assert ev_factor.kind is Factor.Kind.NUMERICAL
        assert ev_factor.expr == 'a'

    def test_repr(self, ev_factor):
        assert repr(ev_factor) == repr(Factor('a'))

    def test_equality(self, ev_factor):
        assert ev_factor == EvaluatedFactor(Factor('a'), [4, 5, 6], kind='numerical')
        assert ev_factor != 'a'

    def test_sort(self, ev_factor):
        assert ev_factor < EvaluatedFactor(Factor('b'), [4, 5, 6], kind='numerical')

        with pytest.raises(TypeError):
            ev_factor < 1

    def test_required_kind(self):
        with pytest.raises(ValueError):
            EvaluatedFactor(Factor('a'), [1, 2, 3])
