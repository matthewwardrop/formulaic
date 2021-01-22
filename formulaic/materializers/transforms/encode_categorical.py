import warnings
from collections import OrderedDict

import numpy
import pandas
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix
from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def encode_categorical(data, reduced_rank=False, spans_intercept=True, output=None, _state=None, _spec=None):
    # TODO: Add support for specifying contrast matrix / etc
    output = output or _spec.output or 'pandas'

    if output == 'sparse':
        data = numpy.array(data)
        data = data.reshape((data.size, ))
        categories, encoded = categorical_encode_series_to_sparse_csc_matrix(data, reduced_rank=reduced_rank)
    else:
        data = pandas.Series(data).astype('category')
        categories = list(data.cat.categories)
        encoded = dict(pandas.get_dummies(data, drop_first=reduced_rank))

    # Update state
    if 'categories' in _state:
        extra_categories = set(categories).difference(_state['categories'])
        if extra_categories:
            warnings.warn(f"Data has categories that were not seen in original dataset: {extra_categories}. This will likely skew the results of your analyses.", DataMismatchWarning)
            for category in extra_categories:
                del encoded[category]

        missing_categories = set(_state['categories']).difference(categories)
        if missing_categories:
            for missing_category in missing_categories:
                if output == 'sparse':
                    encoded[missing_category] = spsparse.csc_matrix((data.shape[0], 1))
                else:
                    encoded[missing_category] = pandas.Series(numpy.zeros(data.shape[0]))
            encoded = OrderedDict(sorted(encoded.items(), key=lambda x: x[0]))
    else:
        _state['categories'] = categories

    encoded.update({
        '__kind__': 'categorical',
        '__spans_intercept__': spans_intercept and not reduced_rank,
        '__drop_field__': _state['categories'][0] if spans_intercept and not reduced_rank else None,
        '__format__': "{name}[T.{field}]",
        '__encoded__': True,
    })

    return encoded
