from functools import singledispatch

from typing import Any

import numpy
import pandas
import scipy.sparse


@singledispatch
def as_columns(data: Any) -> Any:
    """
    Get the columns for `data`. If `data` represents a single column, or is a
    dictionary (the format used to store columns), it is returned as is.
    """
    return data


@as_columns.register
def _(data: pandas.DataFrame):
    return {col: series for col, series in data.items()}


@as_columns.register
def _(data: numpy.ndarray):
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
def _(data: scipy.sparse.csc_matrix):
    if (
        hasattr(data, "__formulaic_metadata__")
        and data.__formulaic_metadata__.column_names
    ):
        column_names = data.__formulaic_metadata__.column_names
    else:
        column_names = list(range(data.shape[1]))
    return {column_names[i]: data[:, i] for i in range(data.shape[1])}
