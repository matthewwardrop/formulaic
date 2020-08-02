import numpy
import scipy.sparse as spsparse

from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def scale(data, center=True, scale=True, ddof=1, state=None):

    data = numpy.array(data)

    if 'ddof' not in state:
        state['ddof'] = ddof
    else:
        ddof = state['ddof']

    # Handle centering
    if 'center' not in state:
        if isinstance(center, bool) and center:
            state['center'] = numpy.mean(data, axis=0)
        elif not isinstance(center, bool):
            state['center'] = numpy.array(center)
        else:
            state['center'] = None
    if state['center'] is not None:
        data = data - state['center']

    # Handle scaling
    if 'scale' not in state:
        if isinstance(scale, bool) and scale:
            state['scale'] = numpy.sqrt(numpy.sum(data ** 2, axis=0) / (data.shape[0] - ddof))
        elif not isinstance(scale, bool):
            state['scale'] = numpy.array(scale)
        else:
            state['scale'] = None
    if state['scale'] is not None:
        data = data / state['scale']

    return data


@scale.register(spsparse.spmatrix)
def _(data, *args, **kwargs):
    assert data.shape[1] == 1
    return scale(data.toarray()[:, 0], *args, **kwargs)


@stateful_transform
def center(data, state=None):
    return scale(data, scale=False, state=state)
