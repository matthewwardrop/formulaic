import numpy
import pandas as pd
import pytest
import scipy.sparse as spsparse

from formulaic import model_matrix
from formulaic.materializers import FactorValues
from formulaic.transforms.hashed import hashed


def _compare_factor_values(a, b, comp=lambda x, y: numpy.allclose(x, y)):
    assert type(a) is type(b)
    if isinstance(a, spsparse.csc_matrix):
        assert comp(
            a.toarray(),
            b,
        )
    else:
        assert comp(a, b)


def test_basic_usage():
    _compare_factor_values(
        hashed(["a", "b", "c"], levels=10),
        FactorValues(
            numpy.array([7, 3, 3]),
            kind="categorical",
            spans_intercept=False,
            column_names=None,
            drop_field=None,
            format="{name}[{field}]",
            encoded=False,
        ),
    )


def test_non_object_input_data():
    _compare_factor_values(
        hashed([1, 2, 35], levels=10),
        FactorValues(
            numpy.array([1, 2, 6]),
            kind="categorical",
            spans_intercept=False,
            column_names=None,
            drop_field=None,
            format="{name}[{field}]",
            encoded=False,
        ),
    )


@pytest.mark.parametrize("levels", [10, 100, 1000])
@pytest.mark.parametrize("dtype", [numpy.int64, numpy.float64, numpy.str_])
def test_usage_in_model_matrix(levels, dtype):
    df = pd.DataFrame({"feature": numpy.array([1, 22, 333]).astype(dtype)})

    m = model_matrix(f"1 + hashed(feature, levels={levels})", df)

    assert m.shape == (3, levels + 1)
    assert m.columns.str.startswith("hashed").sum() == levels
