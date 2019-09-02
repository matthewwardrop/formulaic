from collections import defaultdict

from ..types import Operator, ASTNode, Token


def infix_to_ast(tokens, operators):

    operator_table = defaultdict(dict)
    for operator in operators:
        operator_table[operator.arity][operator.symbol] = operator

    output_queue = []
    operator_stack = []
    last_token = None

    def operate(operator, output_queue):
        return [
            *output_queue[:-operator.arity],
            ASTNode(operator, output_queue[-operator.arity:])
        ]

    for token in tokens:
        if token.kind is not token.Kind.OPERATOR:
            output_queue.append(token)
        else:
            if token.token == '(':
                operator_stack.append(token)
            elif token.token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue = operate(operator_stack.pop(), output_queue)
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop(-1)
            else:
                target_arity = 1 if last_token is None or last_token.kind is Token.Kind.OPERATOR and last_token.token not in ')]' else 2
                operator = operator_table[target_arity][token.token]

                while target_arity == 2 and operator_stack and (operator_stack[-1] != '(') and (
                        operator_stack[-1].precedence > operator.precedence
                        or operator_stack[-1].precedence == operator.precedence and operator.associativity is Operator.Associativity.LEFT
                ):
                    output_queue = operate(operator_stack.pop(), output_queue)

                operator_stack.append(operator)
        last_token = token

    while operator_stack:
        output_queue = operate(operator_stack.pop(), output_queue)

    return output_queue[0]
