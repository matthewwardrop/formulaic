import warnings
from functools import singledispatch

import numpy
import pandas
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix
from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
@singledispatch
def center(data, state=None):
    data = numpy.array(data)
    if 'mean' not in state:
        state['mean'] = numpy.mean(data)
    return data - state['mean']


@center.register(spsparse.spmatrix)
def _(data, state=None):
    assert data.shape[1] == 1
    return center(data.toarray()[:, 0], state=state)


@stateful_transform
@singledispatch
def encode_categorical(data, state, config, reduced_rank=False, spans_intercept=True):
    # TODO: Add support for specifying contrast matrix / etc
    if 'categories' in state:
        data = pandas.Categorical(data, categories=state['categories'])
        if data.shape[0] - data.value_counts().sum():
            warnings.warn("Data has categories that were not seen in original dataset.", DataMismatchWarning)
    else:
        data = pandas.Categorical(data)
        state['categories'] = list(data.categories)
    if config.sparse:
        encoded = categorical_encode_series_to_sparse_csc_matrix(data, reduced_rank=reduced_rank)
    else:
        encoded = dict(pandas.get_dummies(data, drop_first=reduced_rank))
    encoded.update({
        '__kind__': 'categorical',
        '__spans_intercept__': spans_intercept and not reduced_rank,
        '__drop_field__': state['categories'][0] if spans_intercept and not reduced_rank else None,
        '__format__': "{name}[T.{field}]",
        '__encoded__': True,
    })
    return encoded


TRANSFORMS = {
    'center': center,
    'C': encode_categorical,
}
