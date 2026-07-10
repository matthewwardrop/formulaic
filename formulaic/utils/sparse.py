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
    null_indices = numpy.nonzero(series.codes == -1)[0]

    if not levels:
        return levels, spsparse.csc_matrix((series.shape[0], 0))

    codes = series.codes
    if drop_first:
        levels = levels[1:]
        non_null_code_indices = codes > 0
        codes = codes[non_null_code_indices] - 1
    else:
        non_null_code_indices = codes != -1
        codes = codes[non_null_code_indices]
    indices = numpy.arange(series.shape[0])[non_null_code_indices]
    n_columns = len(levels)
    sparse_matrix = spsparse.csc_matrix(
        (
            numpy.ones(codes.shape[0], dtype=float),  # data
            (indices, codes),  # row  # column
        ),
        shape=(series.shape[0], n_columns),
    )
    if null_indices.size:
        # Keep track of null indices (which otherwise would be cast to 0)
        missing_matrix = spsparse.csc_matrix(
            (
                numpy.full(null_indices.size * n_columns, numpy.nan),
                (
                    numpy.tile(null_indices, n_columns),
                    numpy.repeat(numpy.arange(n_columns), null_indices.size),
                ),
            ),
            shape=sparse_matrix.shape,
        )
        sparse_matrix += missing_matrix

    return levels, sparse_matrix
