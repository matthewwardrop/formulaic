import numpy
import pytest
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.materializers.transforms import center


def test_center():
    state = {}
    assert numpy.allclose(center(data=[1, 2, 3], state=state), [-1, 0, 1])
    assert state == {'mean': 2}
    assert numpy.allclose(center(data=[5, 6, 7], state=state), [3, 4, 5])


def test_center_sparse():
    state = {}
    m = spsparse.csc_matrix([1, 2, 3]).transpose()
    assert numpy.allclose(center(data=m, state=state), [-1, 0, 1])
