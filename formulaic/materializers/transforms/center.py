import numpy
import scipy.sparse as spsparse

from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def center(data, state=None):
    data = numpy.array(data)
    if 'mean' not in state:
        state['mean'] = numpy.mean(data)
    return data - state['mean']


@center.register(spsparse.spmatrix)
def _(data, state=None):
    assert data.shape[1] == 1
    return center(data.toarray()[:, 0], state=state)
