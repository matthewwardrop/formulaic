# AST variable extraction
from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Optional, Union

from formulaic.utils.layered_mapping import LayeredMapping


class Variable(str):
    class Role(str, Enum):
        VALUE = "value"
        CALLABLE = "callable"

    roles: set[Role]
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
    def union(cls, *variable_sets: Iterable[Variable]) -> set[Variable]:
        variables: dict[Variable, Variable] = {}
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

    @property
    def root(self) -> Variable:
        """
        Get a `Variable` instance corresponding to the underlying root/data
        variable (e.g. `a.fillna(0)` -> `a`).
        """
        if "." not in self:
            return self
        return Variable(
            self.split(".", 1)[0],
            roles=self.roles.difference({self.Role.CALLABLE}),
            source=self.source,
        )


def get_expression_variables(
    expr: Union[str, ast.AST],
    context: Optional[Mapping] = None,
    aliases: Optional[Mapping] = None,
) -> set[Variable]:
    """
    Extract the variables that are used in the nominated Python expression.

    Args:
        expr: The string or AST representing the python expression.
        context: The context from which variable values will be looked up.
        aliases: A mapping from variable name in the expression to the alias to
            assign to the variable (primarily useful when reverting a variable
            renaming performed during sanitization).
    """
    if isinstance(expr, str):
        expr = ast.parse(expr, mode="eval")
    variables = _get_ast_node_variables(expr, aliases or {})

    if isinstance(context, LayeredMapping):
        out = set()
        for variable in variables:
            variable.source = context.get_layer_name_for_key(variable.split(".", 1)[0])
            out.add(variable)
        return out
    return set(variables)


def _get_ast_node_variables(node: ast.AST, aliases: Mapping) -> list[Variable]:
    variables: list[Variable] = []

    todo = deque([node])
    while todo:
        node = todo.popleft()
        if not isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
            todo.extend(ast.iter_child_nodes(node))
            continue
        name = _get_ast_node_name(node)
        name = aliases.get(name, name)
        if isinstance(node, ast.Call):
            variables.append(Variable(name, roles=["callable"]))
            todo.extend(node.args)
            todo.extend(node.keywords)
        else:
            variables.append(Variable(name, roles=["value"]))

    return variables


def _get_ast_node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _get_ast_node_name(node.func)
    return ast.unparse(node)
