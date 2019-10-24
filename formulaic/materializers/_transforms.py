from functools import singledispatch

import numpy
import pandas

from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix
from formulaic.utils.stateful_transforms import stateful_transform


@singledispatch
@stateful_transform
def center(data, state=None):
    data = numpy.array(data)
    if 'mean' not in state:
        state['mean'] = numpy.mean(data)
    return data - state['mean']


@singledispatch
@stateful_transform
def encode_categorical(data, state, reduced_rank=False, sparse=True):
    # TODO: Add support for specifying contrast matrix / etc
    # TODO: Warn/error when new categories are added to data
    if 'categories' in state:
        data = pandas.Categorical(data, categories=state['categories'])
    else:
        data = pandas.Categorical(data)
        state['categories'] = list(data.categories)
    if sparse:
        return categorical_encode_series_to_sparse_csc_matrix(data, reduced_rank=reduced_rank)
    return dict(pandas.get_dummies(data, drop_first=reduced_rank))


TRANSFORMS = {
    'center': center,
    'C': encode_categorical,
}
