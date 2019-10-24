import ast
import functools

import astor

from .context import LayeredContext


def stateful_transform(func):
    @functools.wraps(func)
    def wrapper(*args, state=None, **kwargs):
        state = {} if state is None else state
        return func(*args, state=state, **kwargs)
    wrapper.__is_stateful_transform__ = True
    return wrapper


def stateful_eval(expr, env, state):
    """
    Evaluate an expression with a given state.

    WARNING: State can be mutated. If you want to preserve a previous state,
    create a copy before passing it to this function.
    """

    state = {} if state is None else state

    # Parse Python code
    code = ast.parse(expr, mode='eval')

    # Extract the nodes of the graph that correspond to stateful transforms
    stateful_nodes = {}
    for node in ast.walk(code):
        if isinstance(node, ast.Call) and getattr(env.get(node.func.id), '__is_stateful_transform__', False):
            stateful_nodes[astor.to_source(node).strip()] = node

    # Mutate stateful nodes to pass in state from a shared dictionary.
    for name, node in stateful_nodes.items():
        name = name.replace('"', r'\\\\"')
        if name not in state:
            state[name] = {}
        node.keywords.append(ast.keyword('state', ast.parse(f'__FORMULAIC_STATE__["{name}"]', mode='eval').body))

    # Compile mutated AST
    code = compile(ast.fix_missing_locations(code), '', 'eval')

    assert "__FORMULAIC_STATE__" not in env

    # Evaluate and return
    return eval(code, {}, LayeredContext({'__FORMULAIC_STATE__': state}, env))
