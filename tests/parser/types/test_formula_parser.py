import pytest

from formulaic.parser.types import FormulaParser


FORMULA_TO_TOKENS = {
    "": [],
    " ": [],
    " \n": [],
    "1": ["1"],
    "a": ["a"],
    "a ~ b": ["a", "~", "b"],
    "a ~": ["a", "~"],
    "~ 1 + a": ["~", "1", "+", "a"],
    "0": ["0"],
    "0 ~": ["0", "~"],
}

PARSER = FormulaParser(operator_resolver=None)


class TestFormulaParser:
    """
    Test the base `FormulaParser` API. We only test the `.get_tokens()` method
    here. See the `DefaultFormulaParser` tests in `../test_parser.py` for a
    richer set of tests.
    """

    @pytest.mark.parametrize("formula,tokens", FORMULA_TO_TOKENS.items())
    def test_get_tokens(self, formula, tokens):
        assert list(PARSER.get_tokens(formula)) == tokens
