import numpy
import scipy.sparse as spsparse

from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def scale(data, center=True, scale=True, ddof=1, _state=None):

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


@scale.register(spsparse.spmatrix)
def _(data, *args, **kwargs):
    assert data.shape[1] == 1
    return scale(data.toarray()[:, 0], *args, **kwargs)


@stateful_transform
def center(data, _state=None):
    return scale(data, scale=False, _state=_state)
