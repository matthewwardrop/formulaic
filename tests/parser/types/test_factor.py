import pytest

from formulaic.parser.types import Factor


class TestFactor:
    @pytest.fixture
    def factor_unknown(self):
        return Factor("unknown")

    @pytest.fixture
    def factor_literal(self):
        return Factor('"string"', kind="literal")

    @pytest.fixture
    def factor_lookup(self):
        return Factor("a", kind="lookup")

    def test_attributes(self):
        assert Factor("a").kind is Factor.Kind.UNKNOWN
        assert Factor("a", kind="constant").kind is Factor.Kind.CONSTANT

        assert Factor("a").eval_method is Factor.EvalMethod.UNKNOWN
        assert Factor("a", eval_method="lookup").eval_method is Factor.EvalMethod.LOOKUP

    def test_equality(self):
        assert Factor("a") == "a"
        assert Factor("a") != 1
        assert Factor("a", kind="constant") == Factor("a", kind="numerical")
        assert Factor("a", eval_method="literal") == Factor("a", eval_method="lookup")

    def test_sort(self):
        a, b, c = Factor("a"), Factor("b"), Factor("c")

        assert a < b
        assert b < c
        assert a < c

        with pytest.raises(TypeError):
            a < 1

    def test_hash(self):
        assert hash(Factor("a")) == hash("a")

    def test_to_terms(self):
        terms = Factor("a").to_terms()
        assert len(terms) == 1

        term = next(iter(terms))
        assert len(term.factors) == 1
        assert next(iter(term.factors)) == Factor("a")

    def test_repr(self):
        assert repr(Factor("a")) == "a"
