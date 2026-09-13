import re

import numpy
import pytest
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.transforms import center, scale


def test_scale():
    state = {}
    assert numpy.allclose(scale(data=[1, 2, 3], _state=state), [-1, 0, 1])
    assert state == {"center": 2.0, "scale": 1.0, "ddof": 1}
    assert numpy.allclose(scale(data=[5, 6, 7], _state=state), [3, 4, 5])

    state = {}
    assert numpy.allclose(
        scale(data=[1, 2, 3], center=False, _state=state),
        [0.37796447, 0.75592895, 1.13389342],
    )
    assert state == {"center": None, "scale": 2.6457513110645907, "ddof": 1}
    assert numpy.allclose(
        scale(data=[5, 6, 7], center=False, _state=state),
        [1.88982237, 2.26778684, 2.64575131],
    )

    state = {}
    assert numpy.allclose(
        scale(data=[1, 2, 3], center=False, scale=False, _state=state), [1, 2, 3]
    )
    assert state == {"center": None, "scale": None, "ddof": 1}

    state = {}
    assert numpy.allclose(
        scale(data=[1, 2, 3], center=1, scale=False, _state=state), [0, 1, 2]
    )
    assert state == {"center": 1, "scale": None, "ddof": 1}

    state = {}
    assert numpy.allclose(
        scale(data=[1, 2, 3], center=1, scale=2, _state=state), [0, 0.5, 1]
    )
    assert state == {"center": 1, "scale": 2, "ddof": 1}

    # Make sure we raise for sparse matrices with more than one column
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot scale a sparse matrix with more than one column."),
    ):
        scale(data=spsparse.csc_matrix([[1, 2, 3], [4, 5, 6]]), center=[1, 2, 3])


def test_center():
    state = {}
    assert numpy.allclose(center(data=[1, 2, 3], _state=state), [-1, 0, 1])
    assert state == {"center": 2, "scale": None, "ddof": 1}
    assert numpy.allclose(center(data=[5, 6, 7], _state=state), [3, 4, 5])


def test_center_sparse():
    state = {}
    m = spsparse.csc_matrix([1, 2, 3]).transpose()
    assert numpy.allclose(center(data=m, _state=state), [-1, 0, 1])


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize("ddof", [0, 1])
def test_regression_scale_missing_values(centered, ddof):
    data = numpy.array([[1, 2], [numpy.nan, 4], [3, numpy.nan], [5, 8]])
    state = {}
    result = scale(data, center=centered, ddof=ddof, _state=state)
    for column in range(data.shape[1]):
        observed = ~numpy.isnan(data[:, column])
        expected = scale(data[observed, column], center=centered, ddof=ddof)
        numpy.testing.assert_allclose(result[observed, column], expected)
    numpy.testing.assert_array_equal(numpy.isnan(result), numpy.isnan(data))
    prediction = numpy.array([[7, 10], [9, numpy.nan]])
    expected = (prediction - state["center"] if centered else prediction) / state[
        "scale"
    ]
    numpy.testing.assert_allclose(
        scale(prediction, _state=state), expected, equal_nan=True
    )


@pytest.mark.parametrize("transform", ["center", "scale"])
def test_regression_scaling_formula_retains_observed_rows(transform):
    import pandas

    from formulaic import model_matrix

    data = pandas.DataFrame({"x": [1, numpy.nan, 2, 3]})
    matrix = model_matrix(f"{transform}(x)", data)
    assert list(matrix.index) == [0, 2, 3]
    numpy.testing.assert_allclose(matrix.iloc[:, 1], [-1, 0, 1])
