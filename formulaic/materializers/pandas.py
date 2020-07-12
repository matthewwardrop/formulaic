import functools
import itertools
from collections import OrderedDict

import numpy
import pandas
import scipy.sparse as spsparse
from interface_meta import override

from .base import FormulaMaterializer
from .types import NAAction


class PandasMaterializer(FormulaMaterializer):

    REGISTRY_NAME = 'pandas'
    DEFAULT_FOR = ['pandas.core.frame.DataFrame']

    def _init(self, sparse=False):
        self.config.sparse = sparse

    @override
    def _is_categorical(self, values):
        if isinstance(values, pandas.Series):
            return values.dtype == object or isinstance(values.dtype, pandas.CategoricalDtype)
        return super()._is_categorical(values)

    @override
    def _check_for_nulls(self, name, values, na_action, drop_rows):

        if na_action is NAAction.IGNORE:
            return

        if isinstance(values, dict):
            for key, vs in values.items():
                self._check_for_nulls(f"{name}[{key}]", vs, na_action, drop_rows)

        elif na_action is NAAction.RAISE:
            if isinstance(values, pandas.Series) and values.isnull().values.any():
                raise ValueError(f"`{name}` contains null values after evaluation.")

        elif na_action is NAAction.DROP:
            if isinstance(values, pandas.Series):
                drop_rows.update(numpy.flatnonzero(values.isnull().values))

        else:
            raise ValueError(f"Do not know how to interpret `na_action` = {repr(na_action)}.")  # pragma: no cover; this is currently impossible to reach

    @override
    def _encode_constant(self, value, metadata, encoder_state, drop_rows):
        if self.config.sparse:
            return spsparse.csc_matrix(numpy.array([value] * self.nrows).reshape((self.nrows - len(drop_rows), 1)))
        return value * pandas.Series(numpy.ones(self.nrows - len(drop_rows)))

    @override
    def _encode_numerical(self, values, metadata, encoder_state, drop_rows):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        if self.config.sparse:
            return spsparse.csc_matrix(numpy.array(values).reshape((self.nrows, 1)))
        return values

    @override
    def _encode_categorical(self, values, metadata, encoder_state, drop_rows, reduced_rank=False):
        # Even though we could reduce rank here, we do not, so that the same
        # encoding can be cached for both reduced and unreduced rank. The
        # rank will be reduced in the _encode_evaled_factor method.
        from .transforms import encode_categorical
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        return encode_categorical(values, metadata=metadata, state=encoder_state, config=self.config, reduced_rank=False)

    @override
    def _get_columns_for_term(self, factors, scale=1):
        out = OrderedDict()

        # Pre-multiply factors with only one set of values (improves performance)
        solo_factors = {}
        indices = []
        for i, factor in enumerate(factors):
            if len(factor) == 1:
                solo_factors.update(factor)
                indices.append(i)
        if solo_factors:
            for index in reversed(indices):
                factors.pop(index)
            if self.config.sparse:
                factors.append({
                    ':'.join(solo_factors): functools.reduce(spsparse.csc_matrix.multiply, solo_factors.values())
                })
            else:
                factors.append({
                    ':'.join(solo_factors): pandas.Series(functools.reduce(lambda a, b: numpy.multiply(a, b), (p for p in solo_factors.values())))
                })

        for product in itertools.product(*(factor.items() for factor in factors)):
            if self.config.sparse:
                out[':'.join(p[0] for p in product)] = scale * functools.reduce(spsparse.csc_matrix.multiply, (p[1] for p in product))
            else:
                out[':'.join(p[0] for p in product)] = scale * functools.reduce(lambda a, b: numpy.multiply(a, b), (p[1].values for p in product))
        return out

    @override
    def _combine_columns(self, cols):
        if self.config.sparse:
            return spsparse.hstack([
                col[1] for col in cols
            ])
        return pandas.concat(
            [
                pandas.Series(col[1], name=col[0])
                for col in cols
            ],
            axis=1,
            copy=False,
        )
