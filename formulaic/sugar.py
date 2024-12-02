from typing import Any, Mapping, Optional, Set, Union

from .formula import FormulaSpec
from .model_matrix import ModelMatrices, ModelMatrix
from .model_spec import ModelSpec, ModelSpecs
from .utils.context import capture_context


def model_matrix(
    spec: Union[FormulaSpec, ModelMatrix, ModelMatrices, ModelSpec, ModelSpecs],
    data: Any,
    *,
    context: Union[int, Mapping[str, Any]] = 0,
    drop_rows: Optional[Set[int]] = None,
    **spec_overrides: Any,
) -> Union[ModelMatrix, ModelMatrices]:
    """
    Generate a model matrix directly from a formula or model spec.

    This method is syntactic sugar for:
    ```
    Formula(
        spec,
        context={"__formulaic_variables_available__": ...},  # used for the `.` operator
    ).get_model_matrix(data, context=LayeredMapping(locals(), globals()), **kwargs)
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
        drop_rows: An optional set of row indices to drop from the model matrix.
            If specified, it will also be updated during materialization with
            any additional rows dropped due to null values.
        spec_overrides: Any `ModelSpec` attributes to set/override. See
            `ModelSpec` for more details.

    Returns:
        The data transformed in to the model matrix with the requested
        nominated structure.
    """
    _context = capture_context(context + 1) if isinstance(context, int) else context
    _spec_context = (  # use materializer context for parser context
        ModelSpec.from_spec([], **spec_overrides)
        .get_materializer(data, context=_context)
        .layered_context
    )

    return ModelSpec.from_spec(
        spec, context=_spec_context, **spec_overrides
    ).get_model_matrix(data, context=_context, drop_rows=drop_rows)
