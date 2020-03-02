import sys

from .formula import Formula
from .utils.layered_mapping import LayeredMapping


def model_matrix(formula, data, *, context=0, **kwargs):
    if isinstance(context, int):
        if hasattr(sys, '_getframe'):
            frame = sys._getframe(context + 1)
            context = LayeredMapping(frame.f_locals, frame.f_globals)
        else:
            context = None  # pragma: no cover
    return Formula(formula).get_model_matrix(data, context=context, **kwargs)
