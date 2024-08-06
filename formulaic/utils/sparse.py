from typing import Iterable, List, Optional, Tuple

import numpy
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(
    series: Iterable, levels: Optional[Iterable[str]] = None, drop_first: bool = False
) -> Tuple[List, spsparse.csc_matrix]:
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

    if drop_first:
        series = series.remove_categories(levels[0])
        levels = levels[1:]

    codes = series.codes
    non_null_code_indices = codes != -1
    indices = numpy.arange(series.shape[0])[non_null_code_indices]
    codes = codes[non_null_code_indices]
    sparse_matrix = spsparse.csc_matrix(
        (
            numpy.ones(codes.shape[0], dtype=float),  # data
            (indices, codes),  # row  # column
        ),
        shape=(series.shape[0], len(levels)),
    )
    return levels, sparse_matrix
