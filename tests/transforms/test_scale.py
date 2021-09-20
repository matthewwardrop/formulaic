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


def test_center():
    state = {}
    assert numpy.allclose(center(data=[1, 2, 3], _state=state), [-1, 0, 1])
    assert state == {"center": 2, "scale": None, "ddof": 1}
    assert numpy.allclose(center(data=[5, 6, 7], _state=state), [3, 4, 5])


def test_center_sparse():
    state = {}
    m = spsparse.csc_matrix([1, 2, 3]).transpose()
    assert numpy.allclose(center(data=m, _state=state), [-1, 0, 1])
