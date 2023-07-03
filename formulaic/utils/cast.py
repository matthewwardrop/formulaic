from functools import singledispatch, wraps
from typing import Any, Callable, Dict, Hashable, Union

import numpy
import pandas
import scipy.sparse

from formulaic.materializers.types.factor_values import FactorValues


def propagate_metadata(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(data, *args, **kwargs):  # type: ignore[no-untyped-def]
        evaluated = func(data, *args, **kwargs)
        if isinstance(data, FactorValues):
            return FactorValues(
                evaluated,
                metadata=data.__formulaic_metadata__,
            )
        return evaluated

    return wrapper


@singledispatch
@propagate_metadata
def as_columns(data: Any) -> Any:
    """
    Get the columns for `data`. If `data` represents a single column, or is a
    dictionary (the format used to store columns), it is returned as is.
    """
    return data


@as_columns.register
@propagate_metadata
def _(data: pandas.DataFrame) -> Dict[Hashable, pandas.Series]:
    return dict(data.items())


@as_columns.register
@propagate_metadata
def _(data: numpy.ndarray) -> Union[numpy.ndarray, Dict[Hashable, numpy.ndarray]]:
    if len(data.shape) == 1:
        return data
    if len(data.shape) > 2:
        raise ValueError(
            "Formulaic does not know how to convert numpy arrays with more than "
            "two dimensions into columns."
        )
    if (
        hasattr(data, "__formulaic_metadata__")
        and data.__formulaic_metadata__.column_names
    ):
        column_names = data.__formulaic_metadata__.column_names
    else:
        column_names = list(range(data.shape[1]))
    return {column_names[i]: data[:, i] for i in range(data.shape[1])}


@as_columns.register
@propagate_metadata
def _(data: scipy.sparse.csc_matrix) -> Dict[Hashable, scipy.sparse.spmatrix]:
    if (
        hasattr(data, "__formulaic_metadata__")
        and data.__formulaic_metadata__.column_names
    ):
        column_names = data.__formulaic_metadata__.column_names
    else:
        column_names = list(range(data.shape[1]))
    return {column_names[i]: data[:, i] for i in range(data.shape[1])}
