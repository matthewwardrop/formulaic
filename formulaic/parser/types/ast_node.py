from __future__ import annotations

import functools
import graphlib
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from formulaic.utils.structured import Structured

from .operator import Operator
from .ordered_set import OrderedSet
from .term import Term

ItemType = TypeVar("ItemType")


class ASTNode(Generic[ItemType]):
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

    def to_terms(
        self, *, context: Optional[Mapping[str, Any]] = None
    ) -> Union[OrderedSet[Term], Structured[OrderedSet[Term]], Tuple]:
        """
        Evaluate this AST node and return the resulting set of `Term` instances.

        Note: We use topological evaluation here to avoid recursion issues for
        long formula (exceeding ~700 terms, though this depends on the recursion
        limit set in the interpreter).

        Args:
            context: An optional context mapping that can be used by operators
                to modify their behaviour (e.g. the `.` operator).
        """
        g = graphlib.TopologicalSorter(self.__generate_evaluation_graph())
        g.prepare()

        results: Dict[ASTNode, Any] = {}

        while g.is_active():
            for node in g.get_ready():
                node_args = tuple(
                    (results[arg] if isinstance(arg, ASTNode) else arg.to_terms())
                    for arg in node.args
                )
                if node.operator.structural or not node_args:
                    results[node] = node.operator.to_terms(*node_args, context=context)
                else:
                    results[node] = Structured._merge(
                        *node_args,
                        merger=functools.partial(
                            node.operator.to_terms, context=context
                        ),
                    )
                g.done(node)

        return results[self]

    def __repr__(self) -> str:
        try:
            return f"<ASTNode {self.operator}: {self.args}>"
        except RecursionError:
            return f"<ASTNode {self.operator}: ...>"

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
                (
                    arg.flatten(str_args=str_args)
                    if isinstance(arg, ASTNode)
                    else (str(arg) if str_args else arg)
                )
                for arg in self.args
            ],
        ]

    # Helpers

    def __generate_evaluation_graph(self) -> Dict[ASTNode, List[ASTNode]]:
        nodes_to_parse = [self]
        graph = {}
        while nodes_to_parse:
            node = nodes_to_parse.pop()
            children = [child for child in node.args if isinstance(child, ASTNode)]
            nodes_to_parse.extend(children)
            graph[node] = children
        return graph
