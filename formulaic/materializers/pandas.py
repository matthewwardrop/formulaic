import functools
import itertools
from collections import defaultdict

import numpy
import pandas
import scipy.sparse as spsparse
from interface_meta import override

from .base import FormulaMaterializer


class PandasMaterializer(FormulaMaterializer):

    REGISTRY_NAME = 'pandas'
    DEFAULT_FOR = ['pandas.core.frame.DataFrame']

    @override
    def _init(self, sparse=False):
        """
        sparse (bool): Output to a sparse matrix.
        """
        self.sparse = sparse

    @override
    def _is_categorical(self, values):
        if isinstance(values, pandas.Series):
            return values.dtype == object or isinstance(values.dtype, pandas.CategoricalDtype)
        return False

    @override
    def _encode_constant(self, value):
        if self.sparse:
            return spsparse.csc_matrix(numpy.array([value]*self.nrows).reshape((self.nrows, 1)))
        return value

    @override
    def _encode_numerical(self, values):
        if self.sparse:
            return spsparse.csc_matrix(numpy.array(values).reshape((self.nrows, 1)))
        return values

    @override
    def _encode_categorical(self, values, reduced_rank=False):
        if self.sparse:
            return categorical_encode_series_to_sparse_csc_matrix(values, reduced_rank=reduced_rank)
        return dict(pandas.get_dummies(values, drop_first=reduced_rank))

    @override
    def _get_columns_for_factors(self, factors, scale=1):
        if self.sparse:
            out = {}
            for product in itertools.product(*(factor.items() for factor in factors)):
                out[':'.join(p[0] for p in product)] = scale * functools.reduce(spsparse.csc_matrix.multiply, (p[1] for p in product))
            return out
        return super()._get_columns_for_factors(factors, scale=scale)

    @override
    def _combine_columns(self, cols):
        if self.sparse:
            return spsparse.hstack(list(cols.values()))
        return pandas.DataFrame(cols)


# Utility methods

def categorical_encode_series_to_sparse_csc_matrix(series, reduced_rank=False):
    results = defaultdict(lambda: [])
    for i, value in enumerate(series):
        results[value].append(i)
    if reduced_rank:
        del results[sorted(results)[0]]
    return {
        value: spsparse.csc_matrix(
            (
                numpy.ones(len(indices), dtype=float),  # data
                (
                    indices,  # row
                    numpy.zeros(len(indices), dtype=int)  # column
                )
            ),
            shape=(series.shape[0], 1)
        )
        for value, indices in results.items()
    }
