from typing import Any, Mapping, Union

from .formula import Formula
from .model_matrix import ModelMatrix
from .model_spec import ModelSpec
from .utils.context import capture_context


def model_matrix(
    spec: Union[Formula, str, list, set, tuple, ModelMatrix, ModelSpec],
    data: Any,
    *,
    context: Union[int, Mapping[str, Any]] = 0,
    **kwargs
) -> ModelMatrix:
    """
    Generate a model matrix directly from a formula or model spec.

    This method is syntactic sugar for:
    ```
    Formula(spec).get_model_matrix(data, context=LayeredMapping(locals(), globals()), **kwargs)
    ```
    or
    ```
    model_spec.get_model_matrix(data, context=LayeredMapping(locals(), globals()), **kwargs)
    ```

    Args:
        spec: The spec that describes the structure of the model matrix to be
            generated. This can be either a `ModelMatrix` or `ModelSpec`
            instance (in which case the structure and state associated with the
            `ModelSpec` instance is re-used), or a formula specification or
            instance (in which case the structure is built from scratch).
        data: The raw data to be transformed into a model matrix. This can be
            any of the supported data types, but is typically a
            `pandas.DataFrame` instance.
        context: The context from which variables (and custom transforms/etc)
            should be inherited. When specified as an integer, it is interpreted
            as a frame offset from the caller's frame (i.e. 0, the default,
            means that all variables in the caller's scope should be made
            accessible when interpreting and evaluating formulae). Otherwise, a
            mapping from variable name to value is expected.
        kwargs: Any additional arguments to pass through to the associated
            `.get_model_matrix()` method.

    Returns:
        The data transformed in to the model matrix with the requested
        nominated structure.
    """
    if isinstance(context, int):
        context = capture_context(context + 1)

    if isinstance(spec, ModelMatrix):
        spec = spec.model_spec
    elif not isinstance(spec, ModelSpec):
        spec = Formula(spec)
    return spec.get_model_matrix(data, context=context, **kwargs)
