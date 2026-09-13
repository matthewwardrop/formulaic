import numpy
import pandas
import pytest

from formulaic.transforms import lag

TEST_CASES = [
    ([1, 2, 3, 4], None, [numpy.nan, 1, 2, 3]),
    ([1, 2, 3, 4], 1, [numpy.nan, 1, 2, 3]),
    ([1, 2, 3, 4], 2, [numpy.nan, numpy.nan, 1, 2]),
    ([1, 2, 3, 4], 0, [1, 2, 3, 4]),
    ([1, 2, 3, 4], -1, [2, 3, 4, numpy.nan]),
]


@pytest.mark.parametrize("data, k, expected", TEST_CASES)
def test_lag_series(data, k, expected):
    assert numpy.allclose(
        lag(pandas.Series(data), k).values
        if k is not None
        else lag(pandas.Series(data)).values,
        expected,
        equal_nan=True,
    )


@pytest.mark.parametrize("data, k, expected", TEST_CASES)
def test_lag_array(data, k, expected):
    assert numpy.allclose(
        lag(numpy.array(data), k) if k is not None else lag(numpy.array(data)),
        expected,
        equal_nan=True,
    )


def test_lag_invalid_type():
    with pytest.raises(NotImplementedError):
        lag(None)


@pytest.mark.parametrize("factory", [numpy.array, pandas.Series])
@pytest.mark.parametrize("k", [-1, 0, 2])
def test_regression_lag_keyword(factory, k):
    data = factory([1, 2, 3, 4])
    numpy.testing.assert_allclose(lag(data, k=k), lag(data, k), equal_nan=True)
    numpy.testing.assert_allclose(lag(data, offset=k), lag(data, k), equal_nan=True)


def test_regression_lag_keyword_narwhals():
    import narwhals.stable.v1 as nw
    import polars

    data = nw.from_native(polars.Series("x", [1, 2, 3]), series_only=True)
    assert lag(data, k=-1).to_list() == [2, 3, None]


def test_regression_lag_keyword_formula():
    from formulaic import model_matrix

    matrix = model_matrix("lag(x, k=2)", pandas.DataFrame({"x": [1, 2, 3, 4]}))
    numpy.testing.assert_array_equal(matrix, [[1, 1], [1, 2]])
