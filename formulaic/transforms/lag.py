from __future__ import annotations

import functools
from typing import Any

import numpy
import pandas


@functools.singledispatch
def lag(data: Any, k: int = 1) -> Any:
    """
    Shift the data indices forward or backward by the specified number of
    intervals.

    Args:
        data: The array-like dataset to be lagged along its first axis.
        k: The number of intervals to shift back the observations by. For "lead"
            shifts, use negative integers.

    Notes:
      * For this transform to make sense, `data` must represent a time-series
        collected at regularly spaced intervals.
      * `k` is chosen as the offset parameter for compatibility with R.
    """
    raise NotImplementedError(
        f"No implementation of `shift` for data of type {type(data)}."
    )


@lag.register
def _(data: pandas.Series, offset: int = 1) -> pandas.Series:
    return data.shift(offset)


@lag.register
def _(data: numpy.ndarray, offset: int = 1) -> numpy.ndarray:
    if data.shape == () or offset == 0:
        return data

    data = numpy.copy(data).astype(float)

    if offset > 0:
        data[offset:] = data[:-offset]
        data[:offset] = numpy.nan
    else:
        data[:offset] = data[-offset:]
        data[offset:] = numpy.nan

    return data
