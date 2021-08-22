from typing import Tuple, Type, Union

from formulaic.errors import FormulaSyntaxError
from .types.ast_node import ASTNode
from .types.token import Token


def exc_for_token(
    token: Union[Token, ASTNode],
    message: str,
    errcls: Type[Exception] = FormulaSyntaxError,
) -> Exception:
    """
    Return an exception ready to be raised with a helpful token/source context.

    Args:
        token: The `Token` or `ASTNode` instance about which an exception should
            be raised.
        message: The message to be included in the exception.
        errcls: The type of the exception to be returned.
    """
    token = __get_token_for_ast(token)
    token_context = token.get_source_context(colorize=True)
    if token_context:
        return errcls(f"{message}\n\n{token_context}")
    return errcls(message)


def exc_for_missing_operator(
    lhs: Union[Token, ASTNode],
    rhs: Union[Token, ASTNode],
    errcls: Type[Exception] = FormulaSyntaxError,
):
    """
    Return an exception ready to be raised about a missing operator token
    between the `lhs` and `rhs` tokens/ast-nodes.

    Args:
        lhs: The `Token` or `ASTNode` instance to the left of where an operator
            should be placed.
        rhs: The `Token` or `ASTNode` instance to the right of where an operator
            should be placed.
        errcls: The type of the exception to be returned.
    """
    lhs_token, rhs_token, error_token = __get_tokens_for_gap(lhs, rhs)
    raise exc_for_token(
        error_token,
        f"Missing operator between `{lhs_token.token}` and `{rhs_token.token}`.",
        errcls=errcls,
    )


def __get_token_for_ast(ast: Union[Token, ASTNode]) -> Token:
    """
    Ensure that incoming `ast` is a `Token`, or else generate one for debugging
    purposes (note that this token will not be valid `Token` for use other than
    in reporting errors).
    """
    if isinstance(ast, Token):
        return ast
    lhs_token = ast
    while isinstance(lhs_token, ASTNode):
        lhs_token = lhs_token.args[0]
    rhs_token = ast
    while isinstance(rhs_token, ASTNode):
        rhs_token = rhs_token.args[-1]
    return Token(
        token=lhs_token.source[lhs_token.source_start : rhs_token.source_end + 1]
        if lhs_token.source
        else "",
        source=lhs_token.source,
        source_start=lhs_token.source_start,
        source_end=rhs_token.source_end,
    )


def __get_tokens_for_gap(
    lhs: Union[Token, ASTNode], rhs: Union[Token, ASTNode]
) -> Tuple[Token, Token, Token]:
    """
    Ensure that incoming `lhs` and `rhs` objects are `Token`s, or else generate
    some for debugging purposes (note that these tokens will not be valid
    `Token`s for use other than in reporting errors). Three tokens will be
    returned: the left-hand side token, the right-hand-side token, and the
    "middle" token where a new operator/token should be inserted (may not
    be empty depending on context).
    """
    lhs_token = lhs
    while isinstance(lhs_token, ASTNode):
        lhs_token = lhs_token.args[-1]
    rhs_token = rhs or lhs
    while isinstance(rhs_token, ASTNode):
        rhs_token = rhs_token.args[0]
    return (
        lhs_token,
        rhs_token,
        Token(
            lhs_token.source[lhs_token.source_start : rhs_token.source_end + 1]
            if lhs_token.source
            else "",
            source=lhs_token.source,
            source_start=lhs_token.source_start,
            source_end=rhs_token.source_end,
        ),
    )
