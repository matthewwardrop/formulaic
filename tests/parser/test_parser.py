import pytest

from formulaic.parser import FormulaParser


FORMULA_TO_TOKENS = {
    '': ['1'],
    ' ': ['1'],
    ' \n': ['1'],

    # Test insertion of '1 + '
    '1': ['1', '+', '1'],
    'a': ['1', '+', 'a'],
    'a ~ b': ['a', '~', '1', '+', 'b'],
    'a ~': ['a', '~', '1'],
    '~ 1 + a': ['~', '1', '+', '1', '+', 'a'],

    # Test translation of '0'
    '0': ['1', '-', '1'],
    '0 ~': ['0', '~', '1']
}

FORMULA_TO_TERMS = {
    '': ['1'],
    'a': ['1', 'a'],
    'a + b': ['1', 'a', 'b'],

    # Interpretation of set differences
    'a - 1': ['a'],
    'a + 0': ['a'],
    'a + b - a': ['1', 'b'],
    '(a + b) - a': ['1', 'b'],
    '(a - a) + b': ['1', 'b'],
    'a + (b - a)': ['1', 'a', 'b'],

    # Formula separators
    '~ a + b': (['1', 'a', 'b'], ),
    'a ~ b + c': (['a'], ['1', 'b', 'c']),
    'a ~ b ~ c': (['a'], ['b'], ['1', 'c']),

    # Products
    'a:b': ['1', 'a:b'],
    'a * b': ['1', 'a', 'a:b', 'b'],
    '(a+b):(c+d)': ['1', 'a:c', 'a:d', 'b:c', 'b:d'],
    '(a+b)**2': ['1', 'a', 'a:b', 'b'],
    '(a+b)**3': ['1', 'a', 'a:b', 'b'],

    # Nested products
    'a/b': ['1', 'a', 'a:b'],
    '(a+b)/c': ['1', 'a', 'a:b:c', 'b'],
    'a/(b+c)': ['1', 'a', 'a:b', 'a:c'],
    'a/(b+c-b)': ['1', 'a', 'a:c'],

    # Unary operations
    '+1': ['1'],
    '-0': ['1'],
}

PARSER = FormulaParser()


class TestFormulaParser:

    @pytest.mark.parametrize("formula,tokens", FORMULA_TO_TOKENS.items())
    def test_get_tokens(self, formula, tokens):
        assert PARSER.get_tokens(formula) == tokens

    @pytest.mark.parametrize("formula,terms", FORMULA_TO_TERMS.items())
    def test_to_terms(self, formula, terms):
        generated_terms = PARSER.get_terms(formula)
        if isinstance(generated_terms, tuple):
            comp = tuple(
                sorted([str(term) for term in group])
                for group in generated_terms
            )
        else:
            comp = sorted([str(term) for term in generated_terms])
        assert comp == terms
