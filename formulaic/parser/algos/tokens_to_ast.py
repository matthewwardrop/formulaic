from collections import namedtuple
from typing import Iterable, List, Union

from ..types import ASTNode, Operator, OperatorResolver, Token
from ..utils import exc_for_token, exc_for_missing_operator


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

    def stack_operator(operator: Union[Token, Operator], token: Token) -> None:
        operator_stack.append(OrderedOperator(operator, token, len(output_queue)))

    def operate(
        ordered_operator: OrderedOperator, output_queue: List[Union[Token, ASTNode]]
    ) -> List[Union[Token, ASTNode]]:
        operator, token, index = ordered_operator

        if operator.fixity is Operator.Fixity.INFIX:
            assert operator.arity == 2
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
            max_prefix_arity = (
                len(output_queue) - operator_stack[-1].index
                if operator_stack
                else len(output_queue)
            )
            operators = operator_resolver.resolve(
                token,
                max_prefix_arity=max_prefix_arity,
                context=[s.operator for s in operator_stack],
            )

            for operator in operators:

                while (
                    operator_stack
                    and operator_stack[-1].token.kind is not Token.Kind.CONTEXT
                    and (
                        operator_stack[-1].operator.precedence > operator.precedence
                        or operator_stack[-1].operator.precedence == operator.precedence
                        and operator.associativity is Operator.Associativity.LEFT
                    )
                ):
                    output_queue = operate(operator_stack.pop(), output_queue)

                stack_operator(operator, token)
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
            raise exc_for_missing_operator(output_queue[0], output_queue[1])
        return output_queue[0]

    return None
