import pytest

from formulaic.parser.algos.tokenize import tokenize


TOKEN_TESTS = {
    '': [],
    'a': ['name:a'],
    'a+b': ['name:a', 'operator:+', 'name:b'],
    'a + b': ['name:a', 'operator:+', 'name:b'],
    'a * b': ['name:a', 'operator:*', 'name:b'],
    'a * (b + c:d)': ['name:a', 'operator:*', 'operator:(', 'name:b', 'operator:+', 'name:c', 'operator::', 'name:d', 'operator:)'],
    'a() + d(a=1, b=2, c  = 3)': ['python:a()', 'operator:+', 'python:d(a=1, b=2, c  = 3)'],
    '1.32 + "string" / b': ['value:1.32', 'operator:+', 'value:"string"', 'operator:/', 'name:b'],
    'a ++ b': ['name:a', 'operator:++', 'name:b'],
    'a +   + b': ['name:a', 'operator:++', 'name:b'],
}


@pytest.mark.parametrize("formula,tokens", TOKEN_TESTS.items())
def test_tokenize(formula, tokens):
    assert [
        f"{token.kind.value}:{token.token}"
        for token in tokenize(formula)
    ] == tokens
