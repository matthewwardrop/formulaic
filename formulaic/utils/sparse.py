from typing import Iterable, Optional

import numpy
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(
    series: Iterable, levels: Optional[Iterable[str]] = None, drop_first: bool = False
) -> spsparse.csc_matrix:
    """
    Categorically encode (via dummy encoding) a `series` as a sparse matrix.

    Args:
        series: The iterable which should be sparse encoded.
        levels: The levels for which to generate dummies (if not specified, a
            dummy variable is generated for every level in `series`).
        drop_first: Whether to omit the first column in order to avoid
            structural collinearity.

    Returns:
        The sparse (column-major) matrix representation of the series dummy
        encoding.
    """
    df = pandas.DataFrame(
        {"series": pandas.Series(series).astype("category").reset_index(drop=True)}
    )
    levels = list(levels or df.series.cat.categories)
    if drop_first:
        levels = levels[1:]
    results = df.groupby("series").groups.copy()

    return levels, spsparse.hstack(
        [
            spsparse.csc_matrix(
                (
                    numpy.ones(len(indices), dtype=float),  # data
                    (indices, numpy.zeros(len(indices), dtype=int)),  # row  # column
                ),
                shape=(numpy.array(series).shape[0], 1),
            )
            for level in levels
            for indices in (results.get(level, []),)
        ]
    )
