from typing import Any

import numpy
import scipy.sparse as spsparse

from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def scale(  # pylint: disable=dangerous-default-value  # always replaced by stateful-transform
    data: Any,
    center: bool = True,
    scale: bool = True,
    ddof: float = 1,
    _state: dict = {},
) -> numpy.ndarray:
    """
    Rescale `data` by centering and re-scaling it.

    Args:
        center: Whether to center the data (subtract the mean).
        scale: Whether to rescale the data such that the standard deviation is
            1.
        ddof: The delta degrees of freedom (default=1, which is equivalent to
            the Bessel correction).
    """

    data = numpy.array(data)

    if "ddof" not in _state:
        _state["ddof"] = ddof
    else:
        ddof = _state["ddof"]

    # Handle centering
    if "center" not in _state:
        if isinstance(center, bool) and center:
            _state["center"] = numpy.mean(data, axis=0)
        elif not isinstance(center, bool):
            _state["center"] = numpy.array(center)
        else:
            _state["center"] = None
    if _state["center"] is not None:
        data = data - _state["center"]

    # Handle scaling
    if "scale" not in _state:
        if isinstance(scale, bool) and scale:
            _state["scale"] = numpy.sqrt(
                numpy.sum(data**2, axis=0) / (data.shape[0] - ddof)
            )
        elif not isinstance(scale, bool):
            _state["scale"] = numpy.array(scale)
        else:
            _state["scale"] = None
    if _state["scale"] is not None:
        data = data / _state["scale"]

    return data


@scale.register  # type: ignore[attr-defined]
def _(data: spsparse.spmatrix, *args: Any, **kwargs: Any) -> numpy.ndarray:
    if data.shape[1] != 1:
        raise ValueError("Cannot scale a sparse matrix with more than one column.")
    return scale(data.toarray()[:, 0], *args, **kwargs)


@stateful_transform
def center(  # pylint: disable=dangerous-default-value  # always replaced by stateful-transform
    data: Any, _state: dict = {}
) -> numpy.ndarray:
    """
    Centers the data by subtracting the mean.
    """
    return scale(data, scale=False, _state=_state)
