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

    REGISTER_NAME = "pandas"
    REGISTER_INPUTS = ("pandas.core.frame.DataFrame",)
    REGISTER_OUTPUTS = ("pandas", "numpy", "sparse")

    @override
    def _is_categorical(self, values):
        if isinstance(values, (pandas.Series, pandas.Categorical)):
            return values.dtype == object or isinstance(
                values.dtype, pandas.CategoricalDtype
            )
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
            raise ValueError(
                f"Do not know how to interpret `na_action` = {repr(na_action)}."
            )  # pragma: no cover; this is currently impossible to reach

    @override
    def _encode_constant(self, value, metadata, encoder_state, spec, drop_rows):
        if spec.output == "sparse":
            return spsparse.csc_matrix(
                numpy.array([value] * self.nrows).reshape(
                    (self.nrows - len(drop_rows), 1)
                )
            )
        series = value * numpy.ones(self.nrows - len(drop_rows))
        return series

    @override
    def _encode_numerical(self, values, metadata, encoder_state, spec, drop_rows):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        if spec.output == "sparse":
            return spsparse.csc_matrix(numpy.array(values).reshape((self.nrows, 1)))
        return values

    @override
    def _encode_categorical(
        self, values, metadata, encoder_state, spec, drop_rows, reduced_rank=False
    ):
        # Even though we could reduce rank here, we do not, so that the same
        # encoding can be cached for both reduced and unreduced rank. The
        # rank will be reduced in the _encode_evaled_factor method.
        from formulaic.transforms import encode_categorical

        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        return encode_categorical(
            values,
            reduced_rank=False,
            _metadata=metadata,
            _state=encoder_state,
            _spec=spec,
        )

    @override
    def _get_columns_for_term(self, factors, spec, scale=1):
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
            if spec.output == "sparse":
                factors.append(
                    {
                        ":".join(solo_factors): functools.reduce(
                            spsparse.csc_matrix.multiply, solo_factors.values()
                        )
                    }
                )
            else:
                factors.append(
                    {
                        ":".join(solo_factors): functools.reduce(
                            lambda a, b: numpy.multiply(a, b),
                            (p for p in solo_factors.values()),
                        )
                    }
                )

        for product in itertools.product(*(factor.items() for factor in factors)):
            if spec.output == "sparse":
                out[":".join(p[0] for p in product)] = scale * functools.reduce(
                    spsparse.csc_matrix.multiply, (p[1] for p in product)
                )
            else:
                out[":".join(p[0] for p in product)] = scale * functools.reduce(
                    lambda a, b: numpy.multiply(a, b),
                    (numpy.array(p[1]) for p in product),
                )
        return out

    @override
    def _combine_columns(self, cols, spec, drop_rows):

        # If we are outputing a pandas DataFrame, explicitly override index
        # in case transforms/etc have lost track of it.
        if spec.output == "pandas":
            pandas_index = self.data_context.index
            if drop_rows:
                pandas_index = pandas_index.drop(self.data_context.index[drop_rows])

        # Special case no columns to empty csc_matrix, array, or DataFrame
        if not cols:
            values = numpy.empty((self.data.shape[0], 0))
            if spec.output == "sparse":
                return spsparse.csc_matrix(values)
            elif spec.output == "numpy":
                return values
            else:
                return pandas.DataFrame(index=pandas_index)

        # Otherwise, concatenate columns into model matrix
        if spec.output == "sparse":
            return spsparse.hstack([col[1] for col in cols])
        if spec.output == "numpy":
            return numpy.stack([col[1] for col in cols], axis=1)
        return pandas.DataFrame(
            {col[0]: col[1] for col in cols},
            index=pandas_index,
            copy=False,
        )
