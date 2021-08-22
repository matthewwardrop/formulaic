from typing import Any, Iterable, List

from .operator import Operator
from .term import Term


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
