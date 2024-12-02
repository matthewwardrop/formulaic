import re
from typing import Iterable, Optional, Sequence, Set, Tuple, Type, Union

from formulaic.errors import FormulaSyntaxError

from .types.ast_node import ASTNode
from .types.token import Token

# Exception handling


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
    extra: Optional[str] = None,
) -> Exception:
    """
    Return an exception ready to be raised about a missing operator token
    between the `lhs` and `rhs` tokens/ast-nodes.

    Args:
        lhs: The `Token` or `ASTNode` instance to the left of where an operator
            should be placed.
        rhs: The `Token` or `ASTNode` instance to the right of where an operator
            should be placed.
        errcls: The type of the exception to be returned.
        extra: Any additional information to be included in the exception message.
    """
    lhs_token, rhs_token, error_token = __get_tokens_for_gap(lhs, rhs)
    return exc_for_token(
        error_token,
        f"Missing operator between `{lhs_token.token}` and `{rhs_token.token}`.{f' {extra}' if extra else ''}",
        errcls=errcls,
    )


def __get_token_for_ast(ast: Union[Token, ASTNode]) -> Token:  # pragma: no cover
    """
    Ensure that incoming `ast` is a `Token`, or else generate one for debugging
    purposes (note that this token will not be valid `Token` for use other than
    in reporting errors).
    """
    if isinstance(ast, Token):
        return ast
    lhs_token = ast
    while isinstance(lhs_token, ASTNode):
        lhs_token = lhs_token.args[0]  # type: ignore
    rhs_token = ast
    while isinstance(rhs_token, ASTNode):
        rhs_token = rhs_token.args[-1]  # type: ignore
    return Token(
        token=(
            lhs_token.source[lhs_token.source_start : rhs_token.source_end + 1]
            if lhs_token.source
            else ""
        ),
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
        lhs_token = (
            lhs_token.args[-1]  # type: ignore
            if lhs_token.args
            else Token(lhs_token.operator.symbol)
        )
    rhs_token = rhs or lhs
    while isinstance(rhs_token, ASTNode):
        rhs_token = (
            rhs_token.args[0]  # type: ignore
            if rhs_token.args
            else Token(rhs_token.operator.symbol)
        )
    return (
        lhs_token,
        rhs_token,
        Token(
            (
                lhs_token.source[lhs_token.source_start : rhs_token.source_end + 1]
                if lhs_token.source
                and lhs_token.source_start is not None
                and rhs_token.source_end is not None
                else ""
            ),
            source=lhs_token.source,
            source_start=lhs_token.source_start,
            source_end=rhs_token.source_end,
        ),
    )


# Token sequence mutations


def replace_tokens(
    tokens: Iterable[Token],
    token_to_replace: str,
    replacement: Union[Token, Sequence[Token]],
    *,
    kind: Optional[Token.Kind] = None,
) -> Iterable[Token]:
    """
    Replace any token in the `tokens` sequence with one or more replacement
    tokens.

    Args:
        tokens: The sequence of tokens within which tokens should be replaced.
        token_to_replace: The string representation of the token to replace.
        replacement: The replacement token(s) to insert into the `tokens`
            sequence.
        kind: The type of tokens to be replaced. If not specified, all
            tokens which match the provided `token_to_match` string will be
            replaced.
    """

    for token in tokens:
        if kind and token.kind is not kind or token.token != token_to_replace:
            yield token
        else:
            if isinstance(replacement, Token):
                yield replacement
            else:
                yield from replacement


def insert_tokens_after(
    tokens: Iterable[Token],
    pattern: Union[str, re.Pattern],
    tokens_to_add: Sequence[Token],
    *,
    kind: Optional[Token.Kind] = None,
    join_operator: Optional[str] = None,
    no_join_for_operators: Union[bool, Set[str]] = True,
) -> Iterable[Token]:
    """
    Insert additional tokens into a sequence of tokens after (within token)
    pattern matches.

    Note: this insertion can happen in the *middle* of existing tokens, which is
    especially useful when inserting tokens around multiple operators (which are
    often merged together into a single token). If you want to avoid this, make
    sure your regex `pattern` includes start and end matchers; e.g.
    `^<pattern>$`.

    Args:
        tokens: The sequence of tokens within which tokens should be replaced.
        pattern: A (potentially compiled) regex expression indicating where
            tokens should be inserted.
        tokens_to_add: A sequence of tokens to be inserted wherever `pattern`
            matches.
        kind: The type of tokens to be considered for insertion. If not
            specified, any matching token (part) will result in insertions.
        join_operator: If the insertion of tokens would result the joining of
            the added tokens with existing tokens, the value set here will be
            used to create a joining operator token. If not provided, not
            additional operators are added.
        no_join_for_operators: Whether to use the join operator when the next
            token is an operator token; or a set of operator symbols for which
            to skip adding the join token.

    """
    tokens = list(tokens)

    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern)

    for i, token in enumerate(tokens):
        if (
            kind is not None
            and token.kind is not kind
            or not pattern.search(token.token)
        ):
            yield token
            continue

        split_tokens = list(token.split(pattern, after=True))
        for j, split_token in enumerate(split_tokens):
            yield split_token

            m = pattern.search(split_token.token)
            if m and m.span()[1] == len(split_token.token):
                yield from tokens_to_add
                if join_operator:
                    next_token = None
                    if j < len(split_tokens) - 1:
                        next_token = split_tokens[j + 1]
                    elif i < len(tokens) - 1:
                        next_token = tokens[i + 1]
                    if next_token is not None and (
                        next_token.kind is not Token.Kind.OPERATOR
                        or no_join_for_operators is False
                        or isinstance(no_join_for_operators, set)
                        and next_token.token not in no_join_for_operators
                    ):
                        yield Token(join_operator, kind=Token.Kind.OPERATOR)


def merge_operator_tokens(
    tokens: Iterable[Token], symbols: Optional[Set[str]] = None
) -> Iterable[Token]:
    """
    Merge operator tokens within a sequence of tokens.

    This is useful if you have added operator tokens after tokenization, in
    order to allow operator resolution of (e.g.) adjacent `+` and `-` operators.

    Args:
        tokens: The sequence of tokens within which tokens should be replaced.
        symbols: If specified, only adjacent operator symbols appearing within
            this set will be merged.
    """
    pooled_token = None

    for token in tokens:
        if (
            token.kind is not Token.Kind.OPERATOR
            or symbols
            and token.token[0] not in symbols
        ):
            if pooled_token:
                yield pooled_token
                pooled_token = None
            yield token
            continue

        # `token` is an operator that can be collapsed on the left
        if pooled_token:
            pooled_token = token.copy_with_attrs(token=pooled_token.token + token.token)
            if symbols and pooled_token.token[-1] not in symbols:
                yield pooled_token
                pooled_token = None
            continue

        pooled_token = token

    if pooled_token:
        yield pooled_token
