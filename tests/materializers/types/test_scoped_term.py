import pytest

from formulaic.materializers.types import ScopedFactor, ScopedTerm, EvaluatedFactor
from formulaic.parser.types import Factor
from formulaic.utils.variables import Variable


class TestScopedTerm:
    @pytest.fixture
    def scoped_term(self):
        return ScopedTerm(
            [
                ScopedFactor(
                    EvaluatedFactor(Factor("a"), values=[1], variables={Variable("a")})
                ),
                ScopedFactor(
                    EvaluatedFactor(Factor("b"), values=[1], variables={Variable("b")})
                ),
            ]
        )

    @pytest.fixture
    def scoped_term_empty(self):
        return ScopedTerm([], scale=10)

    def test_repr(self, scoped_term, scoped_term_empty):
        assert repr(scoped_term) == "a:b"
        assert repr(scoped_term_empty) == "10*1"

    def test_hash(self, scoped_term, scoped_term_empty):
        assert hash(scoped_term) == hash(("a", "b"))
        assert hash(scoped_term_empty) == hash(())

    def test_equality(self, scoped_term, scoped_term_empty):
        assert scoped_term == scoped_term
        assert scoped_term != scoped_term_empty
        assert scoped_term != 1

    def test_copy(self, scoped_term):
        copied = scoped_term.copy()
        assert copied is not scoped_term
        assert copied.factors == scoped_term.factors
        assert copied.scale == scoped_term.scale

    def test_variables(self, scoped_term):
        assert scoped_term.variables == {"a", "b"}
