from functools import singledispatch
from typing import Any, Sequence, Set, Union

import numpy
import pandas
import scipy.sparse as spsparse

from formulaic.materializers.types import FactorValues


@singledispatch
def find_nulls(values: Any) -> Set[int]:
    """
    Find the indices of rows in `values` that have null/nan values.

    Args:
        values: The values in which to find nulls.
    """
    raise ValueError(
        f"No implementation of `find_nulls()` for type `{repr(type(values))}`."
    )


@find_nulls.register
def _(values: None) -> Set[int]:
    # Literal `None` values have special meaning and are checked elsewhere.
    return set()


@find_nulls.register
def _(values: str) -> Set[int]:
    return set()


@find_nulls.register
def _(values: int) -> Set[int]:
    return _drop_nulls_scalar(values)


@find_nulls.register
def _(values: float) -> Set[int]:
    return _drop_nulls_scalar(values)


def _drop_nulls_scalar(values: Union[int, float]) -> Set[int]:
    if isinstance(values, FactorValues):
        values = values.__wrapped__
    if numpy.isnan(values):
        raise ValueError("Constant value is null, invalidating all rows.")
    return set()


@find_nulls.register
def _(values: list) -> Set[int]:
    if isinstance(values, FactorValues):
        # Older versions of pandas (<1.2) cannot unpack this automatically.
        values = values.__wrapped__
    return find_nulls(pandas.Series(values))


@find_nulls.register
def _(values: dict) -> Set[int]:
    indices = set()
    for vs in values.values():
        indices.update(find_nulls(vs))
    return indices


@find_nulls.register
def _(values: pandas.Series) -> Set[int]:
    return set(numpy.flatnonzero(values.isnull().values))


@find_nulls.register
def _(values: numpy.ndarray) -> Set[int]:
    if len(values.shape) == 0:
        if numpy.isnan(values):
            raise ValueError("Constant value is null, invalidating all rows.")
        return set()

    if len(values.shape) == 1:
        return set(numpy.flatnonzero(numpy.isnan(values)))

    if len(values.shape) == 2:
        return set(numpy.flatnonzero(numpy.any(numpy.isnan(values), axis=1)))

    raise ValueError(
        "Cannot check for null indices for arrays of more than 2 dimensions."
    )


@find_nulls.register
def _(values: spsparse.spmatrix) -> Set[int]:
    rows, _, data = spsparse.find(values)
    null_data_indices = numpy.flatnonzero(numpy.isnan(data))
    return set(rows[null_data_indices])


@singledispatch
def drop_rows(values: Any, indices: Sequence[int]) -> Any:
    """
    Drop rows corresponding to the given indices in `values`.

    Args:
        values: The vector from which to drop rows with the given `indices`.
        indices: The indices of the rows to be dropped.
    """
    raise ValueError(
        f"No implementation of `drop_rows()` for values of type `{repr(type(values))}`."
    )


@drop_rows.register
def _(values: list, indices: Sequence[int]) -> list:
    return [value for i, value in enumerate(values) if i not in indices]


@drop_rows.register
def _(values: pandas.Series, indices: Sequence[int]) -> pandas.Series:
    return values.drop(index=values.index[indices])


@drop_rows.register
def _(values: numpy.ndarray, indices: Sequence[int]) -> numpy.ndarray:
    return numpy.delete(values, indices, axis=0)


@drop_rows.register
def _(values: spsparse.spmatrix, indices: Sequence[int]) -> numpy.ndarray:
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    was_csc = False
    if isinstance(values, spsparse.csc_matrix):
        was_csc = True
        values = values.tocsr()
    indices = list(indices)
    mask = numpy.ones(values.shape[0], dtype=bool)
    mask[indices] = False
    masked = values[mask]

    if was_csc:
        masked = masked.tocsc()

    return masked
