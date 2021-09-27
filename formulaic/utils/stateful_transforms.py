import ast
import functools
import inspect
import keyword
import re
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING

import astor
import numpy

from formulaic.parser.algos.tokenize import tokenize
from formulaic.parser.types import Token
from .layered_mapping import LayeredMapping

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec  # pragma: no cover


def stateful_transform(func: Callable) -> Callable:
    """
    Transform a callable object into a stateful transform.

    This is done by adding special arguments to the callable's signature:
    - _state: The existing state or an empty dictionary.
    - _metadata: Any extra metadata passed about the factor being evaluated.
    - _spec: The `ModelSpec` instance being evaluated (or an empty `ModelSpec`).
    If the callable has any of these in its signature, these will be passed onto
    it; otherwise, they will be swallowed by the stateful transform wrapper.

    Stateful transforms are also transformed into single dispatches, allowing
    different implementations for incoming data types.

    Args:
        func: The function (or other callable) to be made into a stateful
            transform.

    Returns:
        The stateful transform callable.
    """
    func = functools.singledispatch(func)
    params = inspect.signature(func).parameters.keys()

    @functools.wraps(func)
    def wrapper(data, *args, _metadata=None, _state=None, _spec=None, **kwargs):
        from formulaic.model_spec import ModelSpec

        _state = {} if _state is None else _state
        extra_params = {}
        if "_metadata" in params:
            extra_params["_metadata"] = _metadata
        if "_spec" in params:
            extra_params["_spec"] = _spec or ModelSpec([])

        if isinstance(data, dict):
            results = {}
            for key, datum in data.items():
                if isinstance(key, str) and key.startswith("__"):
                    results[key] = datum
                else:
                    statum = _state.get(key, {})
                    results[key] = wrapper(
                        datum, *args, _state=statum, **extra_params, **kwargs
                    )
                    if statum:
                        _state[key] = statum
            return results
        return func(data, *args, _state=_state, **extra_params, **kwargs)

    wrapper.__is_stateful_transform__ = True
    return wrapper


def stateful_eval(
    expr: str,
    env: Optional[Mapping],
    metadata: Optional[Mapping],
    state: Optional[Mapping],
    spec: Optional["ModelSpec"],
) -> Any:
    """
    Evaluate an expression in a nominated environment and with a nominated state.

    Under the hood this calls out to `eval`, and so if incoming expressions are
    not safe, you should make sure that your `env` is properly isolated from
    potentially unsafe methods and/or sys-calls.

    Args:
        expr: The expression to be evaluated.
        env: The environment in which the expression is to be evaluated. This
            environment is the only environment from which variables can be
            looked up during the evaluation.
        metadata: Additional metadata about the expression (passed through to
            stateful transforms).
        state: The current state of any stateful transforms (passed through to
            stateful transforms).
        spec: The current `ModelSpec` instance being evaluated (passed through
            to stateful transforms).

    Returns:
        The result of the evaluation.

    Notes:
        - The state mapping is likely to be mutated in-place when using stateful
            transforms. If you need to retain the original state, copy it
            *before* calling this method.
    """
    metadata = {} if metadata is None else metadata
    state = {} if state is None else state
    env = LayeredMapping(
        env
    )  # We sometimes mutate env, so we make sure we do so in a local mutable layer.

    # Ensure that variable names in code are valid for Python's interpreter
    # If not, create new variable in mutable env layer, and update code.
    expr = sanitize_variable_names(expr, env)

    # Parse Python code
    code = ast.parse(expr, mode="eval")

    # Extract the nodes of the graph that correspond to stateful transforms
    stateful_nodes = {}
    for node in ast.walk(code):
        if _is_stateful_transform(node, env):
            stateful_nodes[astor.to_source(node).strip().replace("\n    ", "")] = node

    # Mutate stateful nodes to pass in state from a shared dictionary.
    for name, node in stateful_nodes.items():
        name = name.replace('"', r'\\\\"')
        if name not in state:
            state[name] = {}
        node.keywords.append(
            ast.keyword(
                "_metadata",
                ast.parse(f'__FORMULAIC_METADATA__.get("{name}")', mode="eval").body,
            )
        )
        node.keywords.append(
            ast.keyword(
                "_state", ast.parse(f'__FORMULAIC_STATE__["{name}"]', mode="eval").body
            )
        )
        node.keywords.append(
            ast.keyword("_spec", ast.parse("__FORMULAIC_SPEC__", mode="eval").body)
        )

    # Compile mutated AST
    code = compile(ast.fix_missing_locations(code), "", "eval")

    assert "__FORMULAIC_METADATA__" not in env
    assert "__FORMULAIC_STATE__" not in env
    assert "__FORMULAIC_SPEC__" not in env

    # Evaluate and return
    return eval(
        code,
        {},
        LayeredMapping(
            {
                "__FORMULAIC_METADATA__": metadata,
                "__FORMULAIC_SPEC__": spec,
                "__FORMULAIC_STATE__": state,
            },
            env,
        ),
    )  # nosec


def sanitize_variable_names(expr: str, env: Mapping) -> str:
    """
    Sanitize any variables names in the expression that are not valid Python
    identifiers.

    This function transforms `expr` into a new expression where identifiers
    that would cause `SyntaxError`s are transformed into valid Python
    identifiers. E.g. `1a` -> `<random string>`. `env` is updated to reflect
    this name change, allowing field names to be any string rather than just
    valid Python names.

    Args:
        expr: The expression to sanitize.
        env: The environment to keep updated with any name substitutions. This
            environment mapping will be mutated in place during this evaluation.

    Returns:
        The sanitized expression.
    """
    tokens = []
    for token in tokenize(expr):
        if token.kind.value == "name":
            name = token.token
            if not name.isidentifier() or keyword.iskeyword(name):
                # Compute recognisable basename
                base_name = "".join(
                    [char if re.match(r"\w", char) else "_" for char in name]
                )
                if base_name[0].isdigit():
                    base_name = "_" + base_name

                # Verify new name is not in env already
                new_name = base_name
                while new_name in env:
                    new_name = (
                        base_name
                        + "_"
                        + "".join(
                            numpy.random.choice(list("abcefghiklmnopqrstuvwxyz"), 10)
                        )
                    )

                # Add new mapping
                env[new_name] = env[name]
                token = Token(new_name, kind="name")
        tokens.append(token)
    return " ".join([str(t) for t in tokens])


def _is_stateful_transform(node: ast.AST, env: Mapping) -> bool:
    """
    Check whether a given ast.Call node enacts a stateful transform given
    the available symbols in `env`.

    Args:
        node: The AST node in question.
        env: The current environment in which the node is evaluated. This is
            used to look up the function handle so it can be inspected.

    Return:
        `True` if the node is a call node and the callable associated with the
        node is a stateful transform. `False` otherwise.
    """
    if not isinstance(node, ast.Call):
        return False

    try:
        func = eval(
            compile(astor.to_source(node.func).strip(), "", "eval"), {}, env
        )  # nosec; Get function handle (assuming it exists in env)
        return getattr(func, "__is_stateful_transform__", False)
    except NameError:
        return False
