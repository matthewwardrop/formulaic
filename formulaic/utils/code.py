import ast
import keyword
import re
import sys
from typing import MutableMapping, Union

import numpy

from .iterators import peekable_iter

# Expression formatting


def format_expr(expr: Union[str, ast.AST]) -> str:  # pragma: no cover; branched code
    if sys.version_info >= (3, 9):
        code = ast.parse(expr, mode="eval") if isinstance(expr, str) else expr
        return ast.unparse(code).replace("\n", " ")

    import astor  # pylint: disable=import-error

    # Note: We use `mode="exec"` here because `astor` inserts parentheses around
    # expressions that cannot be naively removed. We still require that these
    # are `eval`-uable in the `stateful_eval` method.
    code = ast.parse(expr, mode="exec") if isinstance(expr, str) else expr
    return astor.to_source(code).strip().replace("\n    ", "")


# Variable sanitization


UNQUOTED_BACKTICK_MATCHER = re.compile(
    r"(\\\"|\"(?:\\\"|[^\"])*\"|\\'|'(?:\\'|[^'])*'|`)"
)


def sanitize_variable_names(
    expr: str, env: MutableMapping, aliases: MutableMapping, *, template: str = "{}"
) -> str:
    """
    Sanitize any variables names in the expression that are not valid Python
    identifiers and are surrounded by backticks (`). This allows use of field
    names that are not valid Python names.

    This function transforms `expr` into a new expression where identifiers that
    would cause `SyntaxError`s are transformed into valid Python identifiers.
    E.g. "func(`1a`)" -> "func(_1a)". `env` is updated to reflect the mapping of
    the old identifier to the new one, provided that the original variable name
    was already present.

    Args:
        expr: The expression to sanitize.
        env: The environment to keep updated with any name substitutions. This
            environment mapping will be mutated in place during this evaluation.
        aliases: A dictionary/map to update with any variable mappings. Mapping
            is from the sanitized variable back to the original variable.
        template: A template to use for sanitized names, which is mainly useful
            if you need to undo the sanitization by string replacement.

    Returns:
        The sanitized expression.
    """

    expr_parts = peekable_iter(UNQUOTED_BACKTICK_MATCHER.split(expr))

    sanitized_expr = []

    for expr_part in expr_parts:
        if expr_part == "`":
            variable_name_parts = []
            while expr_parts.peek(None) not in ("`", None):
                variable_name_parts.append(next(expr_parts))
            variable_name = "".join(variable_name_parts)
            if expr_parts.peek(None) is None:
                sanitized_expr.append(f"`{variable_name}")
            else:
                next(expr_parts)
                new_name = sanitize_variable_name(variable_name, env, template=template)
                aliases[new_name] = variable_name
                sanitized_expr.append(f" {new_name} ")
        else:
            sanitized_expr.append(expr_part)

    return "".join(sanitized_expr).strip()


def sanitize_variable_name(
    name: str, env: MutableMapping, *, template: str = "{}"
) -> str:
    """
    Generate a valid Python variable name for variable identifier `name`.

    Args:
        name: The variable name to sanitize.
        env: The mapping of variable name to values in the evaluation
            environment. If `name` is present in this mapping, an alias is
            created for the same value for the new variable name.
        template: A template to use for sanitized names, which is mainly useful
            if you need to undo the sanitization by string replacement.
    """
    if name.isidentifier() or keyword.iskeyword(name):
        return name

    # Compute recognisable basename
    base_name = "".join([char if re.match(r"\w", char) else "_" for char in name])
    if base_name[0].isdigit():
        base_name = "_" + base_name

    # Verify new name is not in env already, and if not add a random suffix.
    new_name = template.format(base_name)
    while new_name in env:
        new_name = template.format(
            base_name
            + "_"
            + "".join(numpy.random.choice(list("abcefghiklmnopqrstuvwxyz"), 10))
        )

    # Reuse the value for `name` for `new_name` also.
    if name in env:
        env[new_name] = env[name]

    return new_name
