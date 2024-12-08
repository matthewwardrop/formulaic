from typing import Dict, Iterable

from formulaic.utils.code import format_expr, sanitize_variable_names

from ..types.token import Token


def sanitize_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    """
    Sanitize a sequence of tokens. Given that tokens are user contributed code,
    we need to be able to do various hygiene checks/transforms in order to
    ensure consistent behavior downstream. In particular, we check for:
        - `python` tokens should be consistently formatted so that set operators
            and stateful transforms recognise when tokens are equivalent.
        - possible more in the future
    """
    for token in tokens:
        if token.token == ".":  # noqa: S105
            token.kind = Token.Kind.OPERATOR
        if token.kind is Token.Kind.PYTHON:
            token.token = sanitize_python_code(token.token)
        yield token


def sanitize_python_code(expr: str) -> str:
    """
    Ensure than python code is consistently formatted, and that quoted portions
    (by backticks) are properly handled.
    """
    aliases: Dict[str, str] = {}
    expr = format_expr(
        sanitize_variable_names(expr, {}, aliases, template="_formulaic_{}")
    )
    while aliases:
        alias, orig = aliases.popitem()
        expr = expr.replace(alias, f"`{orig}`")
    return expr
