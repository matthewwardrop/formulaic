import pytest

from formulaic.parser.types import Operator


class TestOperator:
    def test_attributes(self):
        op1 = Operator("~", arity=2, precedence=100)
        assert op1.associativity == Operator.Associativity.NONE
        assert op1.fixity == Operator.Fixity.INFIX

        op2 = Operator(
            "+", arity=2, precedence=100, associativity="left", fixity="infix"
        )
        assert op2.associativity == Operator.Associativity.LEFT
        assert op2.fixity == Operator.Fixity.INFIX

    def test_to_terms(self):
        assert Operator(
            "+", arity=2, precedence=100, to_terms=lambda *args: args
        ).to_terms("a", "b", "c") == (
            "a",
            "b",
            "c",
        )

        with pytest.raises(RuntimeError):
            Operator("+", arity=2, precedence=100).to_terms()

    def test_repr(self):
        assert repr(Operator("op", arity=2, precedence=100)) == "op"
