import re

import pytest

from formulaic.errors import FormulaParsingError, FormulaSyntaxError
from formulaic.parser import FormulaParser, DefaultOperatorResolver
from formulaic.parser.types import Structured, Token


FORMULA_TO_TOKENS = {
    "": ["1"],
    " ": ["1"],
    " \n": ["1"],
    # Test insertion of '1 + '
    "1": ["1", "+", "1"],
    "a": ["1", "+", "a"],
    "a ~ b": ["a", "~", "1", "+", "b"],
    "a ~": ["a", "~", "1"],
    "~ 1 + a": ["~", "1", "+", "1", "+", "a"],
    # Test translation of '0'
    "0": ["1", "-", "1"],
    "0 ~": ["~", "1"],
}

FORMULA_TO_TERMS = {
    "": ["1"],
    "a": ["1", "a"],
    "a + b": ["1", "a", "b"],
    # Interpretation of set differences
    "a - 1": ["a"],
    "a + 0": ["a"],
    "a + b - a": ["1", "b"],
    "(a + b) - a": ["1", "b"],
    "(a - a) + b": ["1", "b"],
    "a + (b - a)": ["1", "a", "b"],
    # Check that "0" -> "-1" substitution works as expected
    "+0": [],
    "(+0)": ["1"],
    "0 + 0": [],
    "0 + 0 + 1": ["1"],
    "0 + 0 + 1 + 0": [],
    "0 - 0": ["1"],
    "0 ~ 0": {"lhs": [], "rhs": []},
    # Formula separators
    "~ a + b": ["1", "a", "b"],
    "a ~ b + c": {"lhs": ["a"], "rhs": ["1", "b", "c"]},
    # Formula parts
    "a | b": (["1", "a"], ["1", "b"]),
    "a | b | c": (["1", "a"], ["1", "b"], ["1", "c"]),
    "a | b ~ c | d": {
        "lhs": (["a"], ["b"]),
        "rhs": (["1", "c"], ["1", "d"]),
    },
    # Products
    "a:b": ["1", "a:b"],
    "a * b": ["1", "a", "a:b", "b"],
    "(a+b):(c+d)": ["1", "a:c", "a:d", "b:c", "b:d"],
    "(a+b)**2": ["1", "a", "a:b", "b"],
    "(a+b)**3": ["1", "a", "a:b", "b"],
    # Nested products
    "a/b": ["1", "a", "a:b"],
    "(a+b)/c": ["1", "a", "a:b:c", "b"],
    "a/(b+c)": ["1", "a", "a:b", "a:c"],
    "a/(b+c-b)": ["1", "a", "a:c"],
    # Unary operations
    "+1": ["1"],
    "-0": ["1"],
    "+x": ["1", "x"],
    "-x": ["1"],
}

PARSER = FormulaParser()


class TestFormulaParser:
    # @pytest.mark.parametrize("formula,tokens", FORMULA_TO_TOKENS.items())
    # def test_get_tokens(self, formula, tokens):
    #     assert PARSER.get_tokens(formula) == tokens

    @pytest.mark.parametrize("formula,terms", FORMULA_TO_TERMS.items())
    def test_to_terms(self, formula, terms):
        generated_terms = PARSER.get_terms(formula)
        if isinstance(generated_terms, Structured):
            comp = generated_terms._map(sorted)._to_dict()
        elif isinstance(generated_terms, tuple):
            comp = tuple(
                sorted([str(term) for term in group]) for group in generated_terms
            )
        else:
            comp = sorted([str(term) for term in generated_terms])
        assert comp == terms

    def test_invalid_formula_separation(selF):
        with pytest.raises(FormulaParsingError):
            PARSER.get_terms("a ~ b ~ c")

    def test_invalid_part_separation(selF):
        with pytest.raises(FormulaParsingError):
            PARSER.get_terms("(a | b)")

    def test_invalid_use_of_zero(self):
        with pytest.raises(FormulaParsingError):
            PARSER.get_terms("a * 0")

    def test_invalid_power(self):
        with pytest.raises(
            FormulaSyntaxError,
            match=r"The right-hand argument of `\*\*` must be a positive integer.",
        ):
            assert PARSER.get_terms("a**b")

        with pytest.raises(
            FormulaSyntaxError,
            match=r"The right-hand argument of `\*\*` must be a positive integer.",
        ):
            assert PARSER.get_terms("a**(b+c)")


class TestDefaultOperatorResolver:
    @pytest.fixture
    def resolver(self):
        return DefaultOperatorResolver()

    def test_resolve(self, resolver):

        assert len(resolver.resolve(Token("+++++"), 1, [])) == 1
        assert resolver.resolve(Token("+++++"), 1, [])[0].symbol == "+"
        assert resolver.resolve(Token("+++++"), 1, [])[0].arity == 2

        assert len(resolver.resolve(Token("+++-+"), 1, [])) == 1
        assert resolver.resolve(Token("+++-+"), 1, [])[0].symbol == "-"
        assert resolver.resolve(Token("+++-+"), 1, [])[0].arity == 2

        assert len(resolver.resolve(Token("*+++-+"), 1, [])) == 2
        assert resolver.resolve(Token("*+++-+"), 1, [])[0].symbol == "*"
        assert resolver.resolve(Token("*+++-+"), 1, [])[0].arity == 2
        assert resolver.resolve(Token("*+++-+"), 1, [])[1].symbol == "-"
        assert resolver.resolve(Token("*+++-+"), 1, [])[1].arity == 1

        with pytest.raises(
            FormulaSyntaxError, match="Operator `/` is incorrectly used."
        ):
            resolver.resolve(Token("*/"), 2, [])

    def test_accepts_context(self, resolver):
        tilde_operator = resolver.resolve(Token("~"), 1, [])[0]

        with pytest.raises(
            FormulaSyntaxError, match=re.escape("Operator `~` is incorrectly used.")
        ):
            resolver.resolve(Token("~"), 1, [tilde_operator])
