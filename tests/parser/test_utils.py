from ntpath import join
import pytest

from formulaic.parser.types import Token
from formulaic.parser.utils import (
    replace_tokens,
    insert_tokens_after,
    merge_operator_tokens,
)


@pytest.fixture
def tokens():
    return [
        Token("1", kind=Token.Kind.VALUE),
        Token("+", kind=Token.Kind.OPERATOR),
        Token("field", kind=Token.Kind.NAME),
    ]


def test_replace_tokens(tokens):
    assert list(replace_tokens(tokens, "+", Token("-"))) == ["1", "-", "field"]
    assert list(replace_tokens(tokens, "+", [Token("-")])) == ["1", "-", "field"]
    assert list(replace_tokens(tokens, "+", Token("-"), kind=Token.Kind.NAME)) == [
        "1",
        "+",
        "field",
    ]


def test_insert_tokens_after(tokens):
    assert list(
        insert_tokens_after(
            tokens,
            r"\+",
            [Token("hi", kind=Token.Kind.NAME)],
            join_operator="-",
        )
    ) == ["1", "+", "hi", "-", "field"]
    assert list(
        insert_tokens_after(
            [
                Token("1", kind=Token.Kind.VALUE),
                Token("+|-", kind=Token.Kind.OPERATOR),
                Token("field", kind=Token.Kind.NAME),
            ],
            r"\|",
            [Token("hi", kind=Token.Kind.NAME)],
            join_operator="+",
        )
    ) == ["1", "+|", "hi", "-", "field"]
    assert list(
        insert_tokens_after(
            [
                Token("1", kind=Token.Kind.VALUE),
                Token("+|", kind=Token.Kind.OPERATOR),
                Token("field", kind=Token.Kind.NAME),
            ],
            r"\|",
            [Token("hi", kind=Token.Kind.NAME)],
            join_operator="+",
        )
    ) == ["1", "+|", "hi", "+", "field"]


def test_merge_operator_tokens(tokens):
    assert list(merge_operator_tokens(tokens)) == ["1", "+", "field"]
    assert list(
        merge_operator_tokens(
            [
                Token("+", kind=Token.Kind.OPERATOR),
                Token("+", kind=Token.Kind.OPERATOR),
                Token("-", kind=Token.Kind.OPERATOR),
            ]
        )
    ) == ["++-"]
    assert list(
        merge_operator_tokens(
            [
                Token("-", kind=Token.Kind.OPERATOR),
                Token("+", kind=Token.Kind.OPERATOR),
                Token("+-", kind=Token.Kind.OPERATOR),
                Token("-", kind=Token.Kind.OPERATOR),
            ],
            symbols={"+"},
        )
    ) == ["-", "++-", "-"]
