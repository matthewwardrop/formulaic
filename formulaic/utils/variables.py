# AST variable extraction
from __future__ import annotations

import ast
from collections import deque
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Set, Union

from formulaic.utils.layered_mapping import LayeredMapping


class Variable(str):
    class Role(str, Enum):
        VALUE = "value"
        CALLABLE = "callable"

    roles: Set[Role]
    source: Optional[str]

    def __new__(
        cls,
        name: str,
        *,
        roles: Optional[Iterable[str]] = None,
        source: Optional[str] = None,
    ) -> Variable:
        s = str.__new__(cls, str(name))
        s.roles = {cls.Role(role) for role in (roles or ())}
        s.source = source
        return s

    @classmethod
    def union(cls, *variable_sets: Set[Variable]) -> Set[Variable]:
        variables: Dict[Variable, Variable] = {}
        for variable_set in variable_sets:
            for variable in variable_set:
                if variable in variables:
                    variables[variable] = Variable(
                        str(variable),
                        roles=variable.roles | variables[variable].roles,
                        source=variable.source,
                    )
                else:
                    variables[variable] = variable
        return set(variables.values())


def get_expression_variables(
    expr: Union[str, ast.AST], context: Mapping
) -> Set[Variable]:
    if isinstance(expr, str):
        expr = ast.parse(expr, mode="eval")
    variables = _get_ast_node_variables(expr)

    if isinstance(context, LayeredMapping):
        out = set()
        for variable in variables:
            variable.source = context.get_layer_name_for_key(variable.split(".", 1)[0])
            out.add(variable)
        return out
    return set(variables)


def _get_ast_node_variables(node: ast.AST) -> List[Variable]:
    variables: List[Variable] = []

    todo = deque([node])
    while todo:
        node = todo.popleft()
        if isinstance(node, ast.Call):
            variables.append(Variable(_get_ast_node_name(node), roles=["callable"]))
            todo.extend(node.args)
            todo.extend(node.keywords)
        elif isinstance(node, (ast.Attribute, ast.Name)):
            variables.append(Variable(_get_ast_node_name(node), roles=["value"]))
        else:
            todo.extend(ast.iter_child_nodes(node))

    return variables


def _get_ast_node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _get_ast_node_name(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_get_ast_node_name(node.value)}.{node.attr}"
    raise ValueError(  # pragma: no cover
        f"Unknown AST node type during variable extraction: {type(node)}. "
        "Please report this!"
    )
