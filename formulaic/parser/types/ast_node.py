from collections import namedtuple

from typing import Any, Iterable, List

from .operator import Operator
from .term import Term
from .token import Token


ASTNodeInContext = namedtuple("ASTNodeInContext", ["parent", "index", "node"])


class ASTNode:
    """
    Represents a node in an Abstract Syntax Tree (AST).

    An `ASTNode` instance is composed of an `Operator` instance and a set of
    arguments to be passed into that operator. The arguments may include nested
    `ASTNode`s or other arguments. Once evaluated, a set of `Term` instances
    is returned.

    Attributes:
        operator: The `Operator` instance associated with this node.
        args: The arguments associated with this node.
    """

    def __init__(self, operator: Operator, args: Iterable[Any]):
        self.operator = operator
        self.args = args

    def to_terms(self) -> Iterable[Term]:
        """
        Evaluate this AST node and return the resulting set of `Term` instances.
        """
        return self.operator.to_terms(*self.args)

    def __repr__(self):
        return f"<ASTNode {self.operator}: {self.args}>"

    def flatten(self, str_args: bool = False) -> List[Any]:
        """
        Flatten this `ASTNode` instance into a list of form: [<operator>, *<args>].

        This is primarily useful during debugging and unit testing, since it
        provides a human readable summary of the entire AST.

        Args:
            str_args: Whether to cast every element of the flattened object to
                a string.
        """
        return [
            str(self.operator) if str_args else self.operator,
            *[
                arg.flatten(str_args=str_args)
                if isinstance(arg, ASTNode)
                else (str(arg) if str_args else arg)
                for arg in self.args
            ],
        ]

    # Mutation helpers

    def walk(self, *, path_filter=None, include_filter=None, parent=None, index=None):
        nodes_to_check = list(
            filter(
                path_filter, [ASTNodeInContext(parent=parent, index=index, node=self)]
            )
        )
        while nodes_to_check:
            current_node = nodes_to_check.pop()
            if not include_filter or include_filter(current_node):
                yield current_node
            if isinstance(current_node.node, ASTNode):
                nodes_to_check.extend(
                    filter(
                        path_filter,
                        [
                            ASTNodeInContext(
                                parent=current_node.node, index=index, node=arg
                            )
                            for index, arg in enumerate(current_node.node.args)
                        ],
                    )
                )

    def replace_arg(self, index, new_child):
        self.args = list(self.args)
        self.args[index] = new_child

    def insert_token(self, token: Token, operator: Operator, before: bool = True):
        parent = self

        if (
            self.operator.precedence > operator.precedence
            or self.operator is not operator
            or operator.associativity is Operator.Associativity.LEFT
        ):
            return ASTNode(operator, (token, self))

        arg = 0 if before else -1
        while isinstance(parent.args[arg], ASTNode) and (
            parent.args[arg].operator.precedence < operator.precedence
            or parent.args[arg].operator is operator
            and operator.associativity is Operator.Associativity.RIGHT
        ):
            print("UPDATING -> ", parent, parent.args[arg])
            parent = parent.args[arg]
        parent.replace_arg(
            arg,
            ASTNode(
                operator,
                (token, parent.args[arg]) if before else (parent.args[arg], token),
            ),
        )
        return self
