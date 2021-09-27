from typing import Iterable

import numpy
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(
    series: Iterable, reduced_rank: bool = False
) -> spsparse.csc_matrix:
    """
    Categorically encode (via dummy encoding) a `series` as a sparse matrix.

    Args:
        series: The iterable which should be sparse encoded.
        reduced_rank: Whether to omit the first column in order to avoid
            structural collinearity.

    Returns:
        The sparse (column-major) matrix representation of the series dummy
        encoding.
    """
    df = pandas.DataFrame({"series": pandas.Categorical(series)})
    results = df.groupby("series").groups
    categories = list(results)
    if reduced_rank:
        del results[sorted(results)[0]]
    return categories, {
        value: spsparse.csc_matrix(
            (
                numpy.ones(len(indices), dtype=float),  # data
                (indices, numpy.zeros(len(indices), dtype=int)),  # row  # column
            ),
            shape=(numpy.array(series).shape[0], 1),
        )
        for value, indices in results.items()
    }
