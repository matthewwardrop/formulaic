from collections.abc import Iterable
from typing import Optional

import numpy
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(
    series: Iterable, levels: Optional[Iterable[str]] = None, drop_first: bool = False
) -> tuple[list, spsparse.csc_matrix]:
    """
    Categorically encode (via dummy encoding) a `series` as a sparse matrix.

    Args:
        series: The iterable which should be sparse encoded.
        levels: The levels for which to generate dummies (if not specified, a
            dummy variable is generated for every level in `series`).
        drop_first: Whether to omit the first column in order to avoid
            structural collinearity.

    Returns:
        A tuple of form `(levels, sparse_matrix)`, where `levels` contains the
        levels that were used to generate dummies, and `sparse_matrix` is the
        sparse (column-major) matrix representation of the series dummy
        encoding.
    """

    series = pandas.Categorical(series, levels)
    levels = list(levels or series.categories)

    if not levels:
        return levels, spsparse.csc_matrix((series.shape[0], 0))

    null_mask = series.codes == -1

    if drop_first:
        series = series.remove_categories(levels[0])
        levels = levels[1:]

    codes = series.codes
    non_null_mask = codes != -1
    indices = numpy.flatnonzero(non_null_mask)
    valid_codes = codes[non_null_mask]
    n_columns = len(levels)

    sparse_matrix = spsparse.csc_matrix(
        (
            numpy.ones(valid_codes.shape[0], dtype=float),  # data
            (indices, valid_codes),  # row  # column
        ),
        shape=(series.shape[0], n_columns),
    )

    if numpy.any(null_mask):
        null_indices = numpy.flatnonzero(null_mask)
        n_null = null_indices.size
        missing_matrix = spsparse.csc_matrix(
            (
                numpy.full(n_null * n_columns, numpy.nan),
                (
                    numpy.tile(null_indices, n_columns),
                    numpy.repeat(numpy.arange(n_columns, dtype=numpy.int32), n_null),
                ),
            ),
            shape=sparse_matrix.shape,
        )
        sparse_matrix += missing_matrix

    return levels, sparse_matrix
