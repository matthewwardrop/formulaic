from formulaic.errors import FormulaSyntaxError
from .types.ast_node import ASTNode
from .types.token import Token


def exc_for_token(token, message, errcls=FormulaSyntaxError):
    token = __get_token_for_ast(token)
    token_context = token.get_source_context(colorize=True)
    if token_context:
        return errcls(f"{message}\n\n{token_context}")
    return errcls(message)


def exc_for_missing_operator(lhs, rhs, errcls=FormulaSyntaxError):
    lhs_token, rhs_token, error_token = __get_tokens_for_gap(lhs, rhs)
    raise exc_for_token(error_token, f"Missing operator between `{lhs_token.token}` and `{rhs_token.token}`.", errcls=errcls)


def __get_token_for_ast(ast):
    if isinstance(ast, Token):
        return ast
    lhs_token = ast
    while isinstance(lhs_token, ASTNode):
        lhs_token = lhs_token.args[0]
    rhs_token = ast
    while isinstance(rhs_token, ASTNode):
        rhs_token = rhs_token.args[-1]
    return Token(
        token=lhs_token.source[lhs_token.source_start:rhs_token.source_end + 1] if lhs_token.source else '',
        source=lhs_token.source, source_start=lhs_token.source_start, source_end=rhs_token.source_end
    )


def __get_tokens_for_gap(lhs, rhs):
    lhs_token = lhs
    while isinstance(lhs_token, ASTNode):
        lhs_token = lhs_token.args[-1]
    rhs_token = rhs or lhs
    while isinstance(rhs_token, ASTNode):
        rhs_token = rhs_token.args[0]
    return lhs_token, rhs_token, Token(
        lhs_token.source[lhs_token.source_start:rhs_token.source_end + 1] if lhs_token.source else '',
        source=lhs_token.source, source_start=lhs_token.source_start, source_end=rhs_token.source_end
    )
