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
