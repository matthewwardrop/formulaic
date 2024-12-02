import re

import pytest

from formulaic.parser.types import Factor, Term
from formulaic.utils.calculus import _differentiate_factors, differentiate_term


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


def test__differentiate_factors():
    t = Term([Factor("a"), Factor("log(a)")])

    assert _differentiate_factors(t.factors, "a", use_sympy=True) == {"(log(a) + 1)"}

    with pytest.raises(
        RuntimeError,
        match=re.escape("Cannot differentiate non-trivial factors without `sympy`."),
    ):
        _differentiate_factors(t.factors, "a", use_sympy=False)
