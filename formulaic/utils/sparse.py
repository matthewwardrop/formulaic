import numpy
import pandas
import scipy.sparse as spsparse


def categorical_encode_series_to_sparse_csc_matrix(series, reduced_rank=False):
    df = pandas.DataFrame({'series': pandas.Categorical(series)})
    results = df.groupby('series').groups
    categories = list(results)
    if reduced_rank:
        del results[sorted(results)[0]]
    return categories, {
        value: spsparse.csc_matrix(
            (
                numpy.ones(len(indices), dtype=float),  # data
                (
                    indices,  # row
                    numpy.zeros(len(indices), dtype=int)  # column
                )
            ),
            shape=(numpy.array(series).shape[0], 1)
        )
        for value, indices in results.items()
    }
