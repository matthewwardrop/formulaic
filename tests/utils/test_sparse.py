import numpy
import pandas
from scipy.sparse import csc_matrix

from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix


def test_sparse_category_encoding():
    data = pandas.Series(list("abcdefgabcdefg"))

    levels, encoded = categorical_encode_series_to_sparse_csc_matrix(data)

    (
        reduced_levels,
        reduced_encoded,
    ) = categorical_encode_series_to_sparse_csc_matrix(data, drop_first=True)

    (
        provided_levels,
        encoded_with_provided_levels,
    ) = categorical_encode_series_to_sparse_csc_matrix(
        data, levels=list("fdb"), drop_first=True
    )

    assert levels == list("abcdefg")
    assert encoded.shape == (14, 7)
    assert numpy.all(encoded.sum(axis=0) == 2 * numpy.ones(7))

    assert reduced_levels == list("bcdefg")
    assert reduced_encoded.shape == (14, 6)
    assert numpy.all(reduced_encoded.sum(axis=0) == 2 * numpy.ones(6))

    numpy.testing.assert_array_equal(encoded.data, numpy.ones(14, dtype=float))
    numpy.testing.assert_array_equal(
        encoded.indices, numpy.array([0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13])
    )
    numpy.testing.assert_array_equal(
        encoded.indptr, numpy.array([0, 2, 4, 6, 8, 10, 12, 14])
    )

    numpy.testing.assert_array_equal(reduced_encoded.data, numpy.ones(12, dtype=float))
    numpy.testing.assert_array_equal(
        reduced_encoded.indices, numpy.array([1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13])
    )
    numpy.testing.assert_array_equal(
        reduced_encoded.indptr, numpy.array([0, 2, 4, 6, 8, 10, 12])
    )

    assert provided_levels == list("db")
    assert encoded_with_provided_levels.shape == (14, 2)
    numpy.testing.assert_array_equal(
        encoded_with_provided_levels.data, numpy.ones(4, dtype=float)
    )
    assert set(encoded_with_provided_levels.indices) == {1, 3, 8, 10}
    numpy.testing.assert_array_equal(
        encoded_with_provided_levels.indptr, numpy.array([0, 2, 4])
    )

    empty_levels, empty_encoded = categorical_encode_series_to_sparse_csc_matrix(
        [], drop_first=True
    )
    assert empty_levels == []
    assert empty_encoded.shape == (0, 0)

    (
        explict_missing_levels,
        explict_missing_encoded,
    ) = categorical_encode_series_to_sparse_csc_matrix([1, 2, 3], levels=[])
    assert explict_missing_levels == []
    assert explict_missing_encoded.shape == (3, 0)
