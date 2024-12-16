from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Set, Tuple, cast

import narwhals.stable.v1 as nw
import numpy
import pandas
import scipy.sparse as spsparse
from interface_meta import override

from formulaic.utils.cast import as_columns
from formulaic.utils.null_handling import drop_rows as drop_nulls
from formulaic.utils.null_handling import find_nulls

from .base import FormulaMaterializer
from .types import NAAction

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
    def _check_for_nulls(
        self, name: str, values: Any, na_action: NAAction, drop_rows: Set[int]
    ) -> None:
        if na_action is NAAction.IGNORE:
            return

        try:
            null_indices = find_nulls(values)

            if na_action is NAAction.RAISE:
                if null_indices:
                    raise ValueError(f"`{name}` contains null values after evaluation.")

            elif na_action is NAAction.DROP:
                drop_rows.update(null_indices)

            else:
                raise ValueError(
                    f"Do not know how to interpret `na_action` = {repr(na_action)}."
                )  # pragma: no cover; this is currently impossible to reach
        except ValueError as e:
            raise ValueError(
                f"Error encountered while checking for nulls in `{name}`: {e}"
            ) from e

    @override
    def _encode_constant(
        self,
        value: Any,
        metadata: Any,
        encoder_state: Dict[str, Any],
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
        encoder_state: Dict[str, Any],
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
        encoder_state: Dict[str, Any],
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
        self, factors: List[Dict[str, Any]], spec: ModelSpec, scale: float = 1
    ) -> Dict[str, Any]:
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
        self, cols: Sequence[Tuple[str, Any]], spec: ModelSpec, drop_rows: Sequence[int]
    ) -> pandas.DataFrame:
        # If we are outputing a pandas DataFrame, explicitly override index
        # in case transforms/etc have lost track of it.
        pandas_index = None
        if spec.output == "pandas" and isinstance(self.data, pandas.DataFrame):
            pandas_index = cast(pandas.DataFrame, self.data).index
            if drop_rows:
                pandas_index = pandas_index.drop(
                    cast(pandas.DataFrame, self.data_context).index[drop_rows]
                )

        # Special case no columns to empty csc_matrix, array, or DataFrame
        if not cols:
            values = numpy.empty((self.data.shape[0], 0))
            if spec.output == "sparse":
                return spsparse.csc_matrix(values)
            if spec.output == "narwhals":
                # TODO: Inconsistent with non-empty case below (where we use to-native)
                return nw.from_native(values, eager_only=True)
            if spec.output == "numpy":
                return values
            return (
                pandas.DataFrame(index=pandas_index)
                if pandas_index is not None
                else pandas.DataFrame(values)
            )

        # Otherwise, concatenate columns into model matrix
        if spec.output == "sparse":
            return spsparse.hstack([col[1] for col in cols])

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
            if pandas_index is not None:
                return df.set_index(pandas_index, drop=True)
            return df
        if spec.output == "numpy":
            return combined.to_numpy()
        raise ValueError(f"Invalid output type: {spec.output}")