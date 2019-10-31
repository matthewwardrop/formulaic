import pytest

import functools
import itertools

from formulaic.parser.algos.infix_to_ast import infix_to_ast
from formulaic.parser.algos.tokenize import tokenize
from formulaic.parser.types import Operator


OPERATORS = [
    Operator("~", arity=2, precedence=-100, associativity=None, to_terms=lambda lhs, rhs: (lhs.to_terms(), rhs.to_terms())),
    Operator("~", arity=1, precedence=-100, associativity=None, fixity='prefix', to_terms=lambda lhs, rhs: (None, rhs.to_terms())),
    Operator("+", arity=2, precedence=100, associativity='left', to_terms=lambda *args: set(itertools.chain(*[arg.to_terms() for arg in args]))),
    Operator("-", arity=2, precedence=100, associativity='left', to_terms=lambda left, right: set(set(left.to_terms()).difference(right.to_terms()))),
    Operator("+", arity=1, precedence=100, associativity='right', fixity='prefix'),
    Operator("-", arity=1, precedence=100, associativity='right', fixity='prefix'),
    Operator("*", arity=2, precedence=200, associativity='left', to_terms=lambda *args: (
        {
            functools.reduce(lambda x, y: x * y, term)
            for term in itertools.product(*[arg.to_terms() for arg in args])
        }
        .union(itertools.chain(*[arg.to_terms() for arg in args]))
    )),
    Operator("/", arity=2, precedence=200, associativity='left'),
    Operator(":", arity=2, precedence=300, associativity='left', to_terms=lambda *args: {
        functools.reduce(lambda x, y: x * y, term)
        for term in itertools.product(*[arg.to_terms() for arg in args])
    }),
    Operator("**", arity=2, precedence=500, associativity='right'),
]


FORMULA_TO_AST_TESTS = {
    '': None,
    ' ': None,
    ' \n': None,

    # Token passthrough
    '1': '1',
    'a': 'a',

    # Simple addition
    '1 + 2': ['+', '1', '2'],
    'a + 1': ['+', 'a', '1'],

    # Parentheses
    '(a + 1)': ['+', 'a', '1'],
    '(((a)))': 'a',

    # Order of operations
    'a + b + 1': ['+', ['+', 'a', 'b'], '1'],
    '(a + (b + 1))': ['+', 'a', ['+', 'b', '1']],

    # Python token parsing
    'np.log(a) + np.cos(b)': ['+', 'np.log(a)', 'np.cos(b)'],

    # LHS / RHS separator
    '~ a': ['~', 'a'],
    'a ~ b + c': ['~', 'a', ['+', 'b', 'c']],

    # Unary operators
    "-1": ["-", "1"],

    # Check operator precedences
    'a + b ~ c * d': ['~', ['+', 'a', 'b'], ['*', 'c', 'd']],
    'a + b * c': ['+', 'a', ['*', 'b', 'c']],
    'a + b:c': ['+', 'a', [':', 'b', 'c']],
    '(a + b):c': [':', ['+', 'a', 'b'], 'c'],
    'a * b:c': ['*', 'a', [':', 'b', 'c']],
    "a + b / c": ["+", "a", ["/", "b", "c"]],
    '-a**2': ['-', ['**', 'a', '2']],
    '-a:b': ['-', [':', 'a', 'b']],
}


@pytest.mark.parametrize("formula,flattened", FORMULA_TO_AST_TESTS.items())
def test_formula_to_ast(formula, flattened):
    ast = infix_to_ast(tokenize(formula), OPERATORS)
    if flattened is None:
        assert ast is flattened
    else:
        assert ast.flatten(str_args=True) == flattened
