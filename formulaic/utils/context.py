import sys
from typing import Any, Mapping, Optional, Union

from .layered_mapping import LayeredMapping


def capture_context(
    context: Optional[Union[int, Mapping[str, Any]]] = 0,
) -> Optional[Mapping[str, Any]]:
    """
    Explicitly capture the context to be used by subsequent formula
    materialisations.

    Note: This function is primarily useful in libraries that wrap Formulaic,
    allowing them to easily decouple the extraction of evaluation context from
    the actual materializations calls, which may be several frames removed from
    the users. Also note that implementers are free to do context collection
    without this method, since passing of a dictionary context will always be
    supported; however using this method allows users to treat formulaic as a
    black box.

    Args:
        context: The context from which variables (and custom transforms/etc)
            should be inherited. When specified as an integer, it is interpreted
            as a frame offset from the caller's frame (i.e. 0, the default,
            means that all variables in the caller's scope should be made
            accessible when interpreting and evaluating formulae). Otherwise, a
            mapping from variable name to value is expected. When nesting in a
            library, and attempting to capture user-context, make sure you
            account for the extra frames introduced by your wrappers.

    Returns:
        The context that should be later passed to the Formulaic materialization
        procedure like: `.get_model_matrix(..., context=<this object>)`.
    """
    if isinstance(context, int):
        if hasattr(sys, "_getframe"):
            frame = sys._getframe(context + 1)
            context = LayeredMapping(frame.f_locals, frame.f_globals)
        else:
            context = None  # pragma: no cover
    return context
