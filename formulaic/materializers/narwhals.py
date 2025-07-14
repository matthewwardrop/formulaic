# pragma: no cover; TODO: experimental

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as nw
import numpy
import pandas
import scipy.sparse as spsparse
from interface_meta import override

from formulaic.utils.cast import as_columns
from formulaic.utils.null_handling import drop_rows as drop_nulls

from .base import FormulaMaterializer

if TYPE_CHECKING:  # pragma: no cover
    from formulaic.model_spec import ModelSpec


class NarwhalsMaterializer(FormulaMaterializer):
    REGISTER_NAME = "narwhals"
    REGISTER_INPUTS: Sequence[str] = (
        "narwhals.DataFrame",
        "narwhals.stable.v1.DataFrame",
    )
    REGISTER_OUTPUTS: Sequence[str] = ("narwhals", "pandas", "numpy", "sparse")

    @override
    @classmethod
    def SUPPORTS_INPUT(cls, data: Any) -> bool:
        return nw.dependencies.is_into_dataframe(data)

    @override
    def _init(self) -> None:
        self.__narwhals_data = nw.from_native(self.data, eager_only=True)
        self.__data_context = self.__narwhals_data.to_dict()

    @override  # type: ignore
    @property
    def data_context(self):
        return self.__data_context

    @override
    def _is_categorical(self, values: Any) -> bool:
        if nw.dependencies.is_narwhals_series(values):
            if not values.dtype.is_numeric():
                return True
        return super()._is_categorical(values)

    @override
    def _encode_constant(
        self,
        value: Any,
        metadata: Any,
        encoder_state: dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
    ) -> Any:
        nrows = self.nrows - len(drop_rows)
        if spec.output == "sparse":
            return spsparse.csc_matrix(numpy.array([value] * nrows).reshape((nrows, 1)))
        series = value * numpy.ones(nrows)
        return series

    @override
    def _encode_numerical(
        self,
        values: Any,
        metadata: Any,
        encoder_state: dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
    ) -> Any:
        if drop_rows:
            values = drop_nulls(values, indices=drop_rows)
        if spec.output == "sparse":
            return spsparse.csc_matrix(
                numpy.array(values).reshape((values.shape[0], 1))
            )
        return values

    @override
    def _encode_categorical(
        self,
        values: Any,
        metadata: Any,
        encoder_state: dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
        reduced_rank: bool = False,
    ) -> Any:
        # Even though we could reduce rank here, we do not, so that the same
        # encoding can be cached for both reduced and unreduced rank. The
        # rank will be reduced in the _encode_evaled_factor method.
        from formulaic.transforms import encode_contrasts

        if drop_rows:
            values = drop_nulls(values, indices=drop_rows)
        if nw.dependencies.is_narwhals_series(values):
            values = values.to_pandas()

        return as_columns(
            encode_contrasts(
                values,
                reduced_rank=False,
                output="pandas" if spec.output == "narwhals" else spec.output,
                _metadata=metadata,
                _state=encoder_state,
                _spec=spec,
            )
        )

    @override
    def _get_columns_for_term(
        self, factors: list[dict[str, Any]], spec: ModelSpec, scale: float = 1
    ) -> dict[str, Any]:
        out = {}

        names = [
            ":".join(reversed(product))
            for product in itertools.product(*reversed(factors))
        ]

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
                            numpy.multiply,
                            (numpy.asanyarray(p) for p in solo_factors.values()),
                        )
                    }
                )

        for i, reversed_product in enumerate(
            itertools.product(*(factor.items() for factor in reversed(factors)))
        ):
            if spec.output == "sparse":
                out[names[i]] = scale * functools.reduce(
                    spsparse.csc_matrix.multiply,
                    (p[1] for p in reversed(reversed_product)),
                )
            else:
                out[names[i]] = scale * functools.reduce(
                    numpy.multiply,
                    (numpy.array(p[1]) for p in reversed(reversed_product)),
                )
        return out

    @override
    def _combine_columns(
        self, cols: Sequence[tuple[str, Any]], spec: ModelSpec, drop_rows: Sequence[int]
    ) -> pandas.DataFrame:
        # Special case no columns to empty csc_matrix, array, or DataFrame
        if not cols:
            values = numpy.empty((self.data.shape[0], 0))
            if spec.output == "sparse":
                return spsparse.csc_matrix(values)
            if spec.output == "narwhals":
                # TODO: This output type is inconsistent with the `.to_native()`
                # below.
                return nw.from_native(pandas.DataFrame(values), eager_only=True)
            if spec.output == "numpy":
                return values
            return pandas.DataFrame(values)

        # Otherwise, concatenate columns into model matrix
        if spec.output == "sparse":
            return spsparse.hstack([col[1] for col in cols], format="csc")

        # TODO: Can we do better than this? Having to reconstitute raw data
        # does not seem ideal.
        combined = nw.from_dict(
            {name: nw.to_native(col, pass_through=True) for name, col in cols},
            native_namespace=nw.get_native_namespace(self.__narwhals_data),
        )
        if spec.output == "narwhals":
            if nw.dependencies.is_narwhals_dataframe(self.data):
                return combined
            return combined.to_native()
        if spec.output == "pandas":
            df = combined.to_pandas()
            return df
        if spec.output == "numpy":
            return combined.to_numpy()
        raise ValueError(f"Invalid output type: {spec.output}")  # pragma: no cover
