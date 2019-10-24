from collections import defaultdict

import numpy
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(series, reduced_rank=False):
    results = defaultdict(lambda: [])
    for i, value in enumerate(series):
        if not pandas.isnull(value):
            results[value].append(i)
    if reduced_rank:
        del results[sorted(results)[0]]
    return {
        value: spsparse.csc_matrix(
            (
                numpy.ones(len(indices), dtype=float),  # data
                (
                    indices,  # row
                    numpy.zeros(len(indices), dtype=int)  # column
                )
            ),
            shape=(series.shape[0], 1)
        )
        for value, indices in results.items()
    }
