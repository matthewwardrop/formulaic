import re

from formulaic.errors import FormulaParsingError

from ..types import Token


def tokenize(formula, word_chars=re.compile(r'[\.\_\w]'), numeric_chars=re.compile(r'[0-9\.]'), whitespace_chars=re.compile(r'\s')):
    quote_context = []
    take = 0

    token = Token(source=formula)

    for i, char in enumerate(formula):
        if take > 0:
            token.update(char, i)
            take -= 1
            continue
        if quote_context and char == '\\':
            token.update(char, i)
            take = 1
            continue
        if quote_context and quote_context[-1] in '}`' and char == quote_context[-1]:
            quote_context.pop(-1)
            yield token
            token = Token(source=formula)
            continue
        if quote_context and char == quote_context[-1]:
            token.update(char, i)
            quote_context.pop(-1)
            if token and not quote_context and token.kind is Token.Kind.PYTHON and char in (']', ')'):
                yield token
                token = Token(source=formula)
            continue
        if quote_context and quote_context[-1] in ('"', "'", "`", "}"):
            token.update(char, i)
            continue

        if char == '{':
            if token:
                yield token
            token = Token(source=formula, kind='python')
            quote_context.append('}')
            continue
        if char == '`':
            if token:
                yield token
            token = Token(source=formula, kind='name')
            quote_context.append('`')
            continue
        if char in '([':
            if token.kind in (Token.Kind.NAME, Token.Kind.PYTHON):
                token.update(char, i, kind=Token.Kind.PYTHON)
                quote_context.append(')' if char == '(' else ']')
            else:
                if token:
                    yield token
                    token = Token(source=formula)
                yield Token(source=formula).update(char, i, kind='operator')
            continue
        if char in ')':
            if token:
                yield token
                token = Token(source=formula)
            yield Token().update(char, i, kind='operator')
            continue

        if token.kind is Token.Kind.PYTHON:
            token.update(char, i)
            continue

        if whitespace_chars.match(char):
            continue

        if char in ('"', "'"):
            if token and token.kind is Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            if not token:
                token.update(char, i, kind='value')
                quote_context.append(char)
            else:
                raise FormulaParsingError(f"Unexpected character {repr(char)} following token '{token}'.")
            continue  # pragma: no cover; workaround bug in coverage

        if word_chars.match(char):
            assert token.kind in (None, Token.Kind.OPERATOR, Token.Kind.VALUE, Token.Kind.NAME), f"Unexpected token kind {token.kind}."
            if token and token.kind is Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            if numeric_chars.match(char) and token.kind in (None, Token.Kind.VALUE):
                kind = 'value'
            else:
                kind = 'name'
            token.update(char, i, kind=kind)
            continue
        else:
            if token and token.kind is not Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            token.update(char, i, kind='operator')
    if token:
        yield token
