import re

import numpy
import pandas
import pytest
import scipy.sparse

from formulaic.materializers.types import FactorValues
from formulaic.utils.null_handling import drop_rows, find_nulls


def test_find_nulls():
    assert find_nulls(None) == set()
    assert find_nulls(FactorValues(None)) == set()
    assert find_nulls(1) == set()
    assert find_nulls(1.0) == set()
    assert find_nulls("string") == set()
    assert find_nulls(numpy.array(1)) == set()
    assert find_nulls([None, 1, 2]) == {0}
    assert find_nulls({"key": [1, 2, numpy.nan], "key2": [numpy.nan, 1, 0]}) == {0, 2}
    assert find_nulls(FactorValues([None, 1, 2])) == {0}
    assert find_nulls(numpy.array([numpy.nan, 1, 2])) == {0}
    assert find_nulls(numpy.array([[numpy.nan, 1], [0, numpy.nan], [1, 1]])) == {0, 1}
    assert find_nulls(FactorValues(numpy.array([numpy.nan, 1, 2]))) == {0}
    assert find_nulls(pandas.Series([None, 1, 2])) == {0}
    assert find_nulls(FactorValues(pandas.Series([None, 1, 2]))) == {0}
    assert find_nulls(scipy.sparse.csc_matrix([[numpy.nan], [1], [2]])) == {0}
    assert find_nulls(
        FactorValues(scipy.sparse.csc_matrix([[numpy.nan], [1], [2]]))
    ) == {0}

    with pytest.raises(
        ValueError, match=re.escape("Constant value is null, invalidating all rows.")
    ):
        find_nulls(numpy.nan)

    with pytest.raises(
        ValueError, match=re.escape("Constant value is null, invalidating all rows.")
    ):
        find_nulls(numpy.array(numpy.nan))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot check for null indices for arrays of more than 2 dimensions."
        ),
    ):
        find_nulls(numpy.array([[[1]]]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "No implementation of `find_nulls()` for type `<class 'tuple'>`."
        ),
    ):
        find_nulls((1, 2, 3))


def test_drop_rows():
    assert drop_rows([1, 2, 3], [1]) == [1, 3]
    assert numpy.all(
        drop_rows(pandas.Series([1, 2, 3]), [0]) == pandas.Series([2, 3], index=[1, 2])
    )
    assert numpy.all(drop_rows(numpy.array([1, 2, 3]), [0]) == numpy.array([2, 3]))
    assert numpy.all(
        drop_rows(scipy.sparse.csc_matrix([[1, 2], [3, 4]]), [0]).todense()
        == numpy.array([[3, 4]])
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "No implementation of `drop_rows()` for values of type `<class 'tuple'>`."
        ),
    ):
        drop_rows((1, 2, 3), ())
