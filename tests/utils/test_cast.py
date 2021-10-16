import numpy
import pandas
import pytest
import scipy.sparse

from formulaic import FactorValues
from formulaic.utils.cast import as_columns


def test_as_columns():
    assert as_columns(1) == 1
    assert as_columns([1, 2, 3, 4]) == [1, 2, 3, 4]

    # Check pandas types
    series = pandas.Series([1, 2, 3])
    assert as_columns(series) is series
    assert list(
        as_columns(pandas.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})).keys()
    ) == ["a", "b"]

    # Check numpy types
    assert numpy.all(as_columns(numpy.array([1, 2, 3])) == numpy.array([1, 2, 3]))
    assert numpy.all(
        as_columns(numpy.array([1, 2, 3]).reshape((-1, 1)))[0] == numpy.array([1, 2, 3])
    )
    assert numpy.all(
        as_columns(
            FactorValues(numpy.array([1, 2, 3]).reshape((-1, 1)), column_names=("a"))
        )["a"]
        == numpy.array([1, 2, 3])
    )

    with pytest.raises(
        ValueError,
        match="Formulaic does not know how to convert numpy arrays with more than two dimensions into columns.",
    ):
        as_columns(numpy.array([1, 2, 3]).reshape((1, 1, 3)))

    # Check sparse types
    assert numpy.all(
        as_columns(scipy.sparse.csc_matrix([[1, 2, 3]]))[0] == numpy.array([1])
    )
    assert numpy.all(
        as_columns(
            FactorValues(
                scipy.sparse.csc_matrix([[1, 2, 3]]), column_names=("a", "b", "c")
            )
        )["a"]
        == numpy.array([1])
    )
