import ast
import functools
import inspect

import astor

from .layered_mapping import LayeredMapping


def stateful_transform(func):
    params = inspect.signature(func).parameters.keys()
    @functools.wraps(func)
    def wrapper(data, *args, state=None, config=None, **kwargs):
        from formulaic.materializers.base import FormulaMaterializer
        state = {} if state is None else state
        if isinstance(config, dict):
            config = FormulaMaterializer.Config(**config)
        else:
            config = config or FormulaMaterializer.Config()
        extra_params = {'config': config} if 'config' in params else {}
        if isinstance(data, dict):
            results = {}
            for key, datum in data.items():
                if isinstance(key, str) and key.startswith('__'):
                    results[key] = datum
                else:
                    statum = state.get(key, {})
                    results[key] = wrapper(datum, *args, state=statum, **extra_params, **kwargs)
                    if statum:
                        state[key] = statum
            return results
        return func(data, *args, state=state, **extra_params, **kwargs)
    wrapper.__is_stateful_transform__ = True
    return wrapper


def stateful_eval(expr, env, state, config):
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
        node.keywords.append(ast.keyword('config', ast.parse(f'__FORMULAIC_CONFIG__', mode='eval').body))

    # Compile mutated AST
    code = compile(ast.fix_missing_locations(code), '', 'eval')

    assert "__FORMULAIC_STATE__" not in env
    assert "__FORMULAIC_CONFIG__" not in env

    # Evaluate and return
    return eval(code, {}, LayeredMapping({'__FORMULAIC_CONFIG__': config, '__FORMULAIC_STATE__': state}, env))  # nosec
