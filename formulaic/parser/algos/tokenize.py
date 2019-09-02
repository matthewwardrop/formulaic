import re

from ..types import Token


WORD_CHAR = re.compile(r'[\.\_\w]')
NUMERIC_CHAR = re.compile(r'[0-9\.]')
WHITESPACE_CHAR = re.compile(r'\s')


def tokenize(formula):
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
        if quote_context and char == quote_context[-1]:
            token.update(char, i)
            quote_context.pop(-1)
            if token and not quote_context and token.kind is Token.Kind.PYTHON and char in (']', ')'):
                yield token
                token = Token(source=formula)
            continue
        if quote_context and quote_context[-1] in ('"', "'"):
            token.update(char, i)
            continue

        if char in '([':
            if token.kind is Token.Kind.NAME:
                token.update(char, i, kind=Token.Kind.PYTHON)
                quote_context.append(')' if char == '(' else ']')
            else:
                if token:
                    yield token
                    token = Token(source=formula)
                yield Token(source=formula).update(char, i, kind='operator')
            continue
        if char == ')':
            if token:
                yield token
                token = Token(source=formula)
            yield Token().update(char, i, kind='operator')
            continue

        if token.kind is Token.Kind.PYTHON:
            token.update(char, i)
            continue

        if WHITESPACE_CHAR.match(char):
            continue

        if char in ('"', "'"):
            if token and token.kind is Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            if not token:
                token.update(char, i, kind='value')
                quote_context.append(char)
            else:
                raise ValueError
            continue

        if WORD_CHAR.match(char):
            if token and token.kind is Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            if not token or token.kind in (Token.Kind.VALUE, Token.Kind.NAME):
                if NUMERIC_CHAR.match(char) and token.kind in (None, Token.Kind.VALUE):
                    kind = 'value'
                else:
                    kind = 'name'
                token.update(char, i, kind=kind)
            else:
                if token:
                    yield token
                token = Token(source=formula)
            continue
        else:
            if token and token.kind is not Token.Kind.OPERATOR:
                yield token
                token = Token(source=formula)
            token.update(char, i, kind='operator')
    if token:
        yield token
