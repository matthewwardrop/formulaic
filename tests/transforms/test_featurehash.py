import numpy
import scipy.sparse as spsparse

from formulaic.materializers import FactorValues
from formulaic.transforms.featurehash import hashed


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
