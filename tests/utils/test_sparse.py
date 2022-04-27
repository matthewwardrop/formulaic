import numpy
import pandas

from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix


def test_sparse_category_encoding():
    data = pandas.Series(list("abcdefgabcdefg"))

    levels, encoded = categorical_encode_series_to_sparse_csc_matrix(data)
    (
        reduced_levels,
        reduced_encoded,
    ) = categorical_encode_series_to_sparse_csc_matrix(data, drop_first=True)

    assert levels == list("abcdefg")
    assert encoded.shape == (14, 7)
    assert numpy.all(encoded.sum(axis=0) == 2 * numpy.ones(7))

    assert reduced_levels == list("bcdefg")
    assert reduced_encoded.shape == (14, 6)
    assert numpy.all(reduced_encoded.sum(axis=0) == 2 * numpy.ones(6))
