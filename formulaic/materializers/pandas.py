import functools
import itertools

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
    def _encode_constant(self, value, encoder_state):
        if self.sparse:
            return spsparse.csc_matrix(numpy.array([value]*self.nrows).reshape((self.nrows, 1)))
        return value

    @override
    def _encode_numerical(self, values, encoder_state):
        if self.sparse:
            return spsparse.csc_matrix(numpy.array(values).reshape((self.nrows, 1)))
        return values

    @override
    def _encode_categorical(self, values, encoder_state, reduced_rank=False):
        from ._transforms import encode_categorical
        return encode_categorical(values, state=encoder_state, reduced_rank=reduced_rank, sparse=self.sparse)

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
