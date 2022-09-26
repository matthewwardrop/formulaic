from typing import Iterable, Optional, Tuple, List

import numpy
import numpy as np
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(
    series: Iterable, levels: Optional[Iterable[str]] = None, drop_first: bool = False
) -> Tuple[List[str], spsparse.csc_matrix]:
    """
    Categorically encode (via dummy encoding) a `series` as a sparse matrix.

    Args:
        series: The iterable which should be sparse encoded.
        levels: The levels for which to generate dummies (if not specified, a
            dummy variable is generated for every level in `series`).
        drop_first: Whether to omit the first column in order to avoid
            structural collinearity.

    Returns:
        A tuple (levels, sparse_matrix), where levels contain levels that were used to generate dummies,
        and sparse_matrix is the sparse (column-major) matrix representation of the series dummy encoding.
    """

    series = pandas.Series(series).astype("category").reset_index(drop=True)
    levels = list(levels or series.cat.categories)
    if drop_first:
        levels = levels[1:]

    series = series.where(series.isin(levels))
    codes, _ = pandas.factorize(series)
    indices = numpy.arange(series.shape[0])[codes != -1]
    codes = codes[codes != -1]
    sparse_matrix = spsparse.csc_matrix(
        (
            numpy.ones(codes.shape[0], dtype=float),  # data
            (indices, codes),  # row  # column
        )
    )
    return levels, sparse_matrix

