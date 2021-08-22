import pytest

import functools
import itertools

from formulaic.errors import FormulaSyntaxError
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.algos.tokens_to_ast import tokens_to_ast
from formulaic.parser.algos.tokenize import tokenize
from formulaic.parser.types import Operator


class ExtendedOperatorResolver(DefaultOperatorResolver):
    @property
    def operators(self):
        return super().operators + [
            # Acts like '+' but as a postfix operator
            Operator(
                "@",
                arity=2,
                precedence=1000,
                associativity="right",
                fixity="postfix",
                to_terms=lambda *args: set(
                    itertools.chain(*[arg.to_terms() for arg in args])
                ),
            ),
        ]


FORMULA_TO_AST_TESTS = {
    "": None,
    " ": None,
    " \n": None,
    # Token passthrough
    "1": "1",
    "a": "a",
    # Simple addition
    "1 + 2": ["+", "1", "2"],
    "a + 1": ["+", "a", "1"],
    "a 1 @": ["@", "a", "1"],
    # Parentheses
    "(a + 1)": ["+", "a", "1"],
    "(((a)))": "a",
    # Order of operations
    "a + b + 1": ["+", ["+", "a", "b"], "1"],
    "(a + (b + 1))": ["+", "a", ["+", "b", "1"]],
    # Python token parsing
    "np.log(a) + np.cos(b)": ["+", "np.log(a)", "np.cos(b)"],
    # LHS / RHS separator
    "~ a": ["~", "a"],
    "a ~ b + c": ["~", "a", ["+", "b", "c"]],
    # Unary operators
    "-1": ["-", "1"],
    # Check operator precedences
    "a + b ~ c * d": ["~", ["+", "a", "b"], ["*", "c", "d"]],
    "a + b * c": ["+", "a", ["*", "b", "c"]],
    "a + b:c": ["+", "a", [":", "b", "c"]],
    "(a + b):c": [":", ["+", "a", "b"], "c"],
    "a * b:c": ["*", "a", [":", "b", "c"]],
    "a + b / c": ["+", "a", ["/", "b", "c"]],
    "-a**2": ["-", ["**", "a", "2"]],
    "-a:b": ["-", [":", "a", "b"]],
}

FORMULA_ERRORS = {
    "a b +": [
        FormulaSyntaxError,
        r"Operator `\+` has insuffient arguments and/or is misplaced.",
    ],
    "( a + b": [FormulaSyntaxError, r"Could not find matching parenthesis."],
    "a + b )": [FormulaSyntaxError, r"Could not find matching parenthesis."],
    "a b": [FormulaSyntaxError, r"Missing operator between `a` and `b`."],
    "y + y2 y3 ~ x + z": [
        FormulaSyntaxError,
        r"Missing operator between `y2` and `y3`.",
    ],
}


@pytest.mark.parametrize("formula,flattened", FORMULA_TO_AST_TESTS.items())
def test_formula_to_ast(formula, flattened):
    ast = tokens_to_ast(tokenize(formula), ExtendedOperatorResolver())
    if flattened is None:
        assert ast is flattened
    else:
        assert ast.flatten(str_args=True) == flattened


@pytest.mark.parametrize("formula,exception_info", FORMULA_ERRORS.items())
def test_tokenize_exceptions(formula, exception_info):
    with pytest.raises(exception_info[0], match=exception_info[1]):
        list(tokens_to_ast(tokenize(formula), ExtendedOperatorResolver()))
