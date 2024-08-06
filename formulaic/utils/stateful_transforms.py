import ast
import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    cast,
)

from .code import format_expr, sanitize_variable_names
from .layered_mapping import LayeredMapping
from .variables import Variable, get_expression_variables

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec  # pragma: no cover


def stateful_transform(func: Callable) -> Callable:
    """
    Transform a callable object into a stateful transform.

    This is done by adding special arguments to the callable's signature:
    - _state: The existing state or an empty dictionary.
    - _metadata: Any extra metadata passed about the factor being evaluated.
    - _spec: The `ModelSpec` instance being evaluated (or an empty `ModelSpec`).
    - _context: A mapping of the name to value for all the variables available
        in the formula evaluation context (including data column names).
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
    params = set(inspect.signature(func).parameters.keys())

    @functools.wraps(func)
    def wrapper(  # type: ignore[no-untyped-def]
        data, *args, _metadata=None, _state=None, _spec=None, _context=None, **kwargs
    ):
        from formulaic.model_spec import ModelSpec

        _state = {} if _state is None else _state
        extra_params = {}
        if "_metadata" in params:
            extra_params["_metadata"] = _metadata
        if "_spec" in params:
            extra_params["_spec"] = _spec or ModelSpec(formula=[])
        if "_context" in params:
            extra_params["_context"] = _context

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

        return func(
            data,
            *args,
            **({"_state": _state} if "_state" in params else {}),
            **extra_params,
            **kwargs,
        )

    wrapper.__is_stateful_transform__ = True  # type: ignore[attr-defined]
    return wrapper


def stateful_eval(
    expr: str,
    env: Optional[Mapping],
    metadata: Optional[Mapping],
    state: Optional[MutableMapping],
    spec: Optional["ModelSpec"],
    variables: Optional[Set[Variable]] = None,
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
        variables: A (optional) set of variables to update with the variables
            used in this stateful evaluation.

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
    aliases: Dict[str, str] = {}
    expr = sanitize_variable_names(expr, env, aliases)

    # Parse Python code
    code = ast.parse(expr, mode="eval")

    if variables is not None:
        variables.update(get_expression_variables(code, env, aliases))

    # Extract the nodes of the graph that correspond to stateful transforms
    stateful_nodes: Dict[str, ast.Call] = {}
    for node in ast.walk(code):
        if _is_stateful_transform(node, env):
            stateful_nodes[format_expr(node)] = cast(ast.Call, node)

    # Mutate stateful nodes to pass in state from a shared dictionary.
    for name, node in stateful_nodes.items():
        name = name.replace('"', r'\\\\"')
        if name not in state:
            state[name] = {}
        node.keywords.append(
            ast.keyword(
                "_context",
                ast.parse("__FORMULAIC_CONTEXT__", mode="eval").body,
            )
        )
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
    compiled = compile(ast.fix_missing_locations(code), "", "eval")

    used_reserved = {
        "__FORMULAIC_CONTEXT__",
        "__FORMULAIC_METADATA__",
        "__FORMULAIC_STATE__",
        "__FORMULAIC_SPEC__",
    }.intersection(env)
    if used_reserved:
        raise RuntimeError(
            f"Reserved names {repr(used_reserved)} are already in use in the "
            "evaluation environment."
        )

    # Evaluate and return
    return eval(
        compiled,
        {},
        LayeredMapping(
            {
                "__FORMULAIC_CONTEXT__": env,
                "__FORMULAIC_METADATA__": metadata,
                "__FORMULAIC_SPEC__": spec,
                "__FORMULAIC_STATE__": state,
            },
            env,
        ),
    )  # nosec


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
        func = eval(compile(format_expr(node.func), "", "eval"), {}, env)  # nosec; Get function handle (assuming it exists in env)
        return getattr(func, "__is_stateful_transform__", False)
    except NameError:
        return False
