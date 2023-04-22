import pytest

from formulaic.parser.types import Factor, Term
from formulaic.utils.calculus import differentiate_term


def test_differentiate_term():
    t = Term([Factor("a"), Factor("log(b)")])

    assert str(differentiate_term(t, ["a"])) == "log(b)"
    assert str(differentiate_term(t, ["log(b)"])) == "a"
    assert str(differentiate_term(t, ["b"])) == "0"


def test_differentiate_term_with_sympy():
    pytest.importorskip("sympy")
    t = Term([Factor("a"), Factor("log(b)")])

    assert str(differentiate_term(t, ["a"], use_sympy=True)) == "log(b)"
    assert (
        str(differentiate_term(t, ["log(b)"], use_sympy=True)) == "0"
    )  # 'log(b)' is not in the free symbol list.
    assert str(differentiate_term(t, ["b"], use_sympy=True)) == "a:(1/b)"
