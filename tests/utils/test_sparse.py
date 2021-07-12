import pandas

from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix


def test_sparse_category_encoding():
    data = pandas.Series(list("abcdefgabcdefg"))

    categories, encoded = categorical_encode_series_to_sparse_csc_matrix(data)
    (
        reduced_categories,
        reduced_encoded,
    ) = categorical_encode_series_to_sparse_csc_matrix(data, reduced_rank=True)

    assert categories == list("abcdefg")
    assert reduced_categories == list("abcdefg")
    assert set(encoded) == set(list("abcdefg"))
    assert set(reduced_encoded) == set(list("bcdefg"))

    for label, col in encoded.items():
        assert col.sum() == 2
        assert col.shape == (14, 1)

    for label, col in reduced_encoded.items():
        assert col.sum() == 2
        assert col.shape == (14, 1)
