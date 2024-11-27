import pytest

from formulaic.errors import FormulaSyntaxError
from formulaic.parser.algos.tokenize import tokenize

TOKEN_TESTS = {
    "": [],
    "a": ["name:a"],
    "a+b": ["name:a", "operator:+", "name:b"],
    "a + b": ["name:a", "operator:+", "name:b"],
    "a * b": ["name:a", "operator:*", "name:b"],
    "a * (b + c:d)": [
        "name:a",
        "operator:*",
        "context:(",
        "name:b",
        "operator:+",
        "name:c",
        "operator::",
        "name:d",
        "context:)",
    ],
    "[a + b]": ["context:[", "name:a", "operator:+", "name:b", "context:]"],
    "a() + d(a=1, b=2, c  = 3)": [
        "python:a()",
        "operator:+",
        "python:d(a=1, b=2, c  = 3)",
    ],
    '1.32 + "string" / b': [
        "value:1.32",
        "operator:+",
        'value:"string"',
        "operator:/",
        "name:b",
    ],
    "a ++ b": ["name:a", "operator:++", "name:b"],
    "a +   + b": ["name:a", "operator:++", "name:b"],
    "a(b() + c())": ["python:a(b() + c())"],
    "len({})": ["python:len({})"],
    r"'\''": [r"value:'\''"],
    '"abc" + "def"': ['value:"abc"', "operator:+", 'value:"def"'],
    "`a|b * 2:a{}`": ["name:a|b * 2:a{}"],
    "{`a|b` @ `b2:1`}": ["python:`a|b` @ `b2:1`"],
    "I(`a|b`)": ["python:I(`a|b`)"],
    "a + `a+b` + {a / b}": [
        "name:a",
        "operator:+",
        "name:a+b",
        "operator:+",
        "python:a / b",
    ],
    "a + `a(` + {a(`a`, '{`', '}`')}": [
        "name:a",
        "operator:+",
        "name:a(",
        "operator:+",
        "python:a(`a`, '{`', '}`')",
    ],
    "a%in%%custom op%": [
        "name:a",
        "operator:in",
        "operator:custom op",
    ],
    "A[T.a]": [
        "python:A[T.a]",
    ],
    "A(T.a)": [
        "python:A(T.a)",
    ],
    "A(T.a)[b](c)": [
        "python:A(T.a)[b](c)",
    ],
    "A(T.a)[b](c)a": ["python:A(T.a)[b](c)", "name:a"],
}

TOKEN_ERRORS = {
    'a"hello"': [FormulaSyntaxError, "Unexpected character '\"' following token `a`."],
    "`a": [
        FormulaSyntaxError,
        "Formula ended before quote context was closed. Expected: `",
    ],
    "A[(T.a]": [
        FormulaSyntaxError,
        r"Formula ended before quote context was closed. Expected: \)",
    ],
    "A([T.a)": [
        FormulaSyntaxError,
        "Formula ended before quote context was closed. Expected: ]",
    ],
}


@pytest.mark.parametrize("formula,tokens", TOKEN_TESTS.items())
def test_tokenize(formula, tokens):
    assert [
        f"{token.kind.value}:{token.token}" for token in tokenize(formula)
    ] == tokens


@pytest.mark.parametrize("formula,exception_info", TOKEN_ERRORS.items())
def test_tokenize_exceptions(formula, exception_info):
    with pytest.raises(exception_info[0], match=exception_info[1]):
        list(tokenize(formula))
