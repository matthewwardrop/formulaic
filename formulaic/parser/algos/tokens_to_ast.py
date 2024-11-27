from collections import namedtuple
from typing import Iterable, List, Set, Union

from ..types import ASTNode, Operator, OperatorResolver, Token
from ..utils import exc_for_missing_operator, exc_for_token

OrderedOperator = namedtuple("OrderedOperator", ("operator", "token", "index"))
CONTEXT_OPENERS = {"(", "["}
CONTEXT_CLOSERS = {
    ")": "(",
    "]": "[",
}


def tokens_to_ast(
    tokens: Iterable[Token], operator_resolver: OperatorResolver
) -> Union[None, Token, ASTNode]:
    """
    Convert a iterable of `Token` instances into an abstract syntax tree.

    This implementation is intentionally as simple and abstract as possible, and
    makes few assumptions about the form of the operators that will be present
    in the token sequence. Instead, it relies on the `OperatorResolver` instance
    to evaluate based on the context which operator should be invoked to handle
    surrounding tokens based on their arity/etc. This means that changes to the
    formula syntax (such as the addition of new operators) should not require
    any changes to this abstract syntax tree generator.

    The algorithm employed here is a slightly enriched [Shunting Yard
    Algorithm](https://en.wikipedia.org/wiki/Shunting-yard_algorithm), where we
    have added additional support for operator arities, fixities,
    associativities, etc.

    Args:
        tokens: The tokens for which an abstract syntax tree should be
            generated.
        operator_resolver: The `OperatorResolver` instance to be used to lookup
            operators (only the `.resolve()` method is used).

    Returns:
        The generated abstract syntax tree as a nested `ASTNode` instance.
    """
    output_queue: List[Union[Token, ASTNode]] = []
    operator_stack: List[OrderedOperator] = []
    disabled_operators: Set[Token] = set()

    def stack_operator(operator: Union[Token, Operator], token: Token) -> None:
        operator_stack.append(OrderedOperator(operator, token, len(output_queue)))

    def operate(
        ordered_operator: OrderedOperator, output_queue: List[Union[Token, ASTNode]]
    ) -> List[Union[Token, ASTNode]]:
        operator, token, index = ordered_operator

        if operator.fixity is Operator.Fixity.INFIX:
            if operator.arity != 2:
                raise exc_for_token(  # pragma: no cover
                    token,
                    f"Infix operator `{token.token}` must have an arity of 2 (got: {operator.arity}).",
                )
            min_index = index - 1
            max_index = index + 1
        elif operator.fixity is Operator.Fixity.PREFIX:
            min_index = index
            max_index = index + operator.arity
        else:  # Operator.Fixity.POSTFIX
            min_index = index - operator.arity
            max_index = index

        if min_index < 0 or max_index > len(output_queue):
            raise exc_for_token(
                token,
                f"Operator `{token.token}` has insuffient arguments and/or is misplaced.",
            )

        return [
            *output_queue[:min_index],
            ASTNode(operator, output_queue[min_index:max_index]),
            *output_queue[max_index:],
        ]

    for token in tokens:
        if token.kind is token.Kind.CONTEXT:
            if token.token in CONTEXT_OPENERS:
                stack_operator(token, token)
            elif token.token in CONTEXT_CLOSERS:
                starting_token = CONTEXT_CLOSERS[token.token]
                while operator_stack and operator_stack[-1].token != starting_token:
                    output_queue = operate(operator_stack.pop(), output_queue)
                if operator_stack and operator_stack[-1].token == starting_token:
                    operator_stack.pop()
                else:
                    raise exc_for_token(
                        token, "Could not find matching context marker."
                    )
            else:  # pragma: no cover
                raise exc_for_token(
                    token,
                    f"Context token `{token.token}` is unrecognized.",
                )
        elif token.kind is token.Kind.OPERATOR:
            for operator_token, operators in operator_resolver.resolve(token):
                for operator in operators:
                    if not operator.accepts_context(
                        [s.operator for s in operator_stack]
                    ):
                        continue
                    if operator.disabled:
                        disabled_operators.add(operator_token)
                        continue
                    # Apply all operators with precedence greater than the current operator
                    while (
                        operator_stack
                        and operator_stack[-1].token.kind is not Token.Kind.CONTEXT
                        and (
                            operator_stack[-1].operator.precedence > operator.precedence
                            or operator_stack[-1].operator.precedence
                            == operator.precedence
                            and operator.associativity is Operator.Associativity.LEFT
                        )
                    ):
                        output_queue = operate(operator_stack.pop(), output_queue)

                    # Determine maximum number of postfix arguments
                    max_postfix_arity = (
                        len(output_queue) - operator_stack[-1].index
                        if operator_stack
                        else len(output_queue)
                    )

                    # Check if operator is valid in current context
                    if (
                        operator.arity == 0
                        or operator.fixity is Operator.Fixity.PREFIX
                        or max_postfix_arity == 1
                        and operator.fixity is Operator.Fixity.INFIX
                        or max_postfix_arity >= operator.arity
                        and operator.fixity is Operator.Fixity.POSTFIX
                    ):
                        stack_operator(operator, token)
                        break
                else:
                    if operator_token in disabled_operators:
                        raise exc_for_token(
                            token,
                            f"Operator `{operator_token}` is at least partially disabled by parser configuration, and/or is incorrectly used.",
                        )
                    raise exc_for_token(
                        token, f"Operator `{operator_token}` is incorrectly used."
                    )
        else:
            output_queue.append(token)

    while operator_stack:
        if operator_stack[-1].token.kind is Token.Kind.CONTEXT:
            raise exc_for_token(
                operator_stack[-1].token, "Could not find matching context marker."
            )
        output_queue = operate(operator_stack.pop(), output_queue)

    if output_queue:
        if len(output_queue) > 1:
            raise exc_for_missing_operator(
                output_queue[0],
                output_queue[1],
                extra=(
                    "This may be due to the following operators being at least "
                    f"partially disabled by parser configuration: {disabled_operators}."
                    if disabled_operators
                    else None
                ),
            )
        return output_queue[0]

    return None
