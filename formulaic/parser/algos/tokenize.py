import re
from typing import Iterable, List, Pattern

from ..types import Token
from ..utils import exc_for_token


def tokenize(
    formula: str,
    word_chars: Pattern = re.compile(r"[\.\_\w]"),
    numeric_chars: Pattern = re.compile(r"[0-9\.]"),
    whitespace_chars: Pattern = re.compile(r"\s"),
) -> Iterable[Token]:
    """
    Convert a formula string into a generator of tokens.

    This tokenizer is intentionally very simple, and it makes no attempt to
    validate incoming tokens beyond ensuring that they are complete. The
    rationale for this is that changes like adding support for a new operator do
    not require changes to this tokenizer, and can instead be done entirely
    within the higher-level parser. This simplicity also lends itself to a direct
    functional implementation (rather than a class with methods), and so that is
    approach taken here.

    Tokens outputted will have one of four kinds:
      - operator: an operator to be applied to other surrounding tokens (will
            always consist of non-word characters).
      - name: a name of a feature/variable to be lifted from the model matrix
            context.
      - value: a literal value (string/number).
      - python: a code string to be evaluated.

    The basic logic of this tokenizer is to loop over each character in the
    formula string and:
      - ensure that portions quoted by one of : ', ", {}, %, and ` are correctly
        grouped into a token of the appropriate kind.
      - ignore unquoted whitespace
      - correctly distinguish users of (, ), [, and ] as grouping operators vs. Python
        function calls.
      - output each contiguous portion of the formula string that belongs to
        the same token type as a token. (e.g. sequential operators like '+-'
        will be output as a single operator token).

    Args:
        formula: The formula string to tokenize.
        word_chars: The regex pattern used to recognize "word" characters
            (basically non-operator characters).
        numeric_chars: The regex pattern used to recognize numeric characters.
        whitespace_chars: The regex pattern use to recognize (ignored)
            whitespace characters.

    Returns:
        A generator over the tokens found in the formula string.

    """
    quote_context: List[str] = []
    take = 0

    token = Token(source=formula)

    for i, char in enumerate(formula):
        if take > 0:
            token.update(char, i)
            take -= 1
            continue
        if quote_context and char == "\\":
            token.update(char, i)
            take = 1
            continue
        if quote_context and quote_context[-1] in "}`%" and char == quote_context[-1]:
            quote_context.pop(-1)
            if token:
                if quote_context:
                    token.update(char, i)
                else:
                    yield token
                    token = Token(source=formula)
            continue
        if quote_context and char == quote_context[-1]:
            token.update(char, i)
            quote_context.pop(-1)
            # if (
            #     token
            #     and not quote_context
            #     and token.kind is Token.Kind.PYTHON
            #     and char in ("]", ")")
            # ):
            #     yield token
            #     token = Token(source=formula)
            continue
        if quote_context and quote_context[-1] in ('"', "'", "`", ")", "]", "}", "%"):
            if char in "`([" and quote_context[-1] in "})]":
                quote_context.append(char.replace("(", ")").replace("[", "]"))
            token.update(char, i)
            continue

        if char == "%":
            if token:
                yield token
            token = Token(source=formula, kind="operator", source_start=i)
            quote_context.append("%")
            continue
        if char == "{":
            if token:
                yield token
            token = Token(source=formula, kind="python", source_start=i)
            quote_context.append("}")
            continue
        if char == "`":
            if token:
                yield token
            token = Token(source=formula, kind="name", source_start=i)
            quote_context.append("`")
            continue
        if char in "([":
            if token.kind in (Token.Kind.NAME, Token.Kind.PYTHON):
                token.update(char, i, kind=Token.Kind.PYTHON)
                quote_context.append(")" if char == "(" else "]")
            else:
                if token:
                    yield token
                    token = Token(source=formula)
                yield Token(source=formula).update(char, i, kind="context")
            continue
        if char in ")]":
            if token:
                yield token
                token = Token(source=formula)
            yield Token(source=formula).update(char, i, kind="context")
            continue

        if whitespace_chars.match(char):
            if token and token.kind is not Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            continue

        if char in ('"', "'"):
            if token and token.kind is Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            if not token:
                token.update(char, i, kind="value")
                quote_context.append(char)
            else:
                raise exc_for_token(
                    Token(source=formula, source_start=i, source_end=i),
                    f"Unexpected character {repr(char)} following token `{token.token}`.",
                )
            continue  # pragma: no cover; workaround bug in coverage

        if word_chars.match(char):
            if token and token.kind in (Token.Kind.OPERATOR, Token.Kind.PYTHON):
                yield token
                token = Token(source=formula)
            if token.kind not in (
                None,
                Token.Kind.VALUE,
                Token.Kind.NAME,
            ):
                raise exc_for_token(  # pragma: no cover
                    Token(source=formula, source_start=i, source_end=i),
                    f"Unexpected token kind {token.kind} for character '{char}'.",
                )
            if numeric_chars.match(char) and token.kind in (None, Token.Kind.VALUE):
                kind = "value"
            else:
                kind = "name"
            token.update(char, i, kind=kind)
            continue
        if token and token.kind is not Token.Kind.OPERATOR:
            yield token
            token = Token(source=formula)
        token.update(char, i, kind="operator")
    if quote_context:
        raise exc_for_token(
            token,
            message=f"Formula ended before quote context was closed. Expected: {quote_context[-1]}",
        )
    if token:
        yield token
