import pytest

from formulaic.parser.algos.infix_to_ast import infix_to_ast


_parser_tests = {
    "": ["~", "1"],
    " ": ["~", "1"],
    " \n ": ["~", "1"],

    "1": ["~", "1"],
    "a": ["~", "a"],
    "a ~ b": ["~", "a", "b"],

    "(a ~ b)": ["~", "a", "b"],
    "a ~ ((((b))))": ["~", "a", "b"],
    "a ~ ((((+b))))": ["~", "a", ["+", "b"]],

    "a + b + c": ["~", ["+", ["+", "a", "b"], "c"]],
    "a + (b ~ c) + d": ["~", ["+", ["+", "a", ["~", "b", "c"]], "d"]],

    "a + np.log(a, base=10)": ["~", ["+", "a", "np.log(a, base=10)"]],
    # Note different spacing:
    "a + np . log(a , base = 10)": ["~", ["+", "a", "np.log(a, base=10)"]],

    # Check precedence
    "a + b ~ c * d": ["~", ["+", "a", "b"], ["*", "c", "d"]],
    "a + b * c": ["~", ["+", "a", ["*", "b", "c"]]],
    "-a**2": ["~", ["-", ["**", "a", "2"]]],
    "-a:b": ["~", ["-", [":", "a", "b"]]],
    "a + b:c": ["~", ["+", "a", [":", "b", "c"]]],
    "(a + b):c": ["~", [":", ["+", "a", "b"], "c"]],
    "a*b:c": ["~", ["*", "a", [":", "b", "c"]]],

    "a+b / c": ["~", ["+", "a", ["/", "b", "c"]]],
    "~ a": ["~", "a"],

    "-1": ["~", ["-", "1"]],
    }

def _compare_trees(got, expected):
    assert isinstance(got, ParseNode)
    if got.args:
        assert got.type == expected[0]
        for arg, expected_arg in zip(got.args, expected[1:]):
            _compare_trees(arg, expected_arg)
    else:
        assert got.type in _atomic_token_types
        assert got.token.extra == expected
