import copy
import pickle

import numpy
import pandas
import pytest

from formulaic import (
    FactorValues,
    Formula,
    ModelMatrices,
    ModelMatrix,
    ModelSpec,
    ModelSpecs,
    model_matrix,
)
from formulaic.formula import OrderingMethod


def test_model_matrix_copy():
    matrix = numpy.array([[1, 2, 3], [4, 5, 6]])
    spec = ModelSpec(formula="x")
    m = ModelMatrix(matrix, spec=spec)

    m2 = copy.copy(m)
    assert numpy.all(m2.__wrapped__ == matrix)
    assert m2.model_spec is spec

    m3 = copy.deepcopy(m)
    assert numpy.all(m3.__wrapped__ == matrix)
    assert m3.model_spec is not spec


def test_model_matrix_pickle():
    matrix = numpy.array([[1, 2, 3], [4, 5, 6]])
    spec = ModelSpec(formula="x")
    m = ModelMatrix(matrix, spec=spec)

    m2 = pickle.loads(pickle.dumps(m))
    assert numpy.all(m2.__wrapped__ == matrix)
    assert m2.model_spec is not spec


def test_factor_values_copy():
    d = {"1": object()}
    f = FactorValues(d, drop_field="test")

    f2 = copy.copy(f)
    assert f2.__wrapped__ == d
    assert f2.__wrapped__["1"] is d["1"]
    assert f2.__formulaic_metadata__.drop_field == "test"

    f3 = copy.deepcopy(f)
    assert f3.__wrapped__["1"] is not d["1"]
    assert f3.__formulaic_metadata__.drop_field == "test"


def test_model_matrices():
    model_matrices = model_matrix(
        "y ~ x", pandas.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
    )
    assert isinstance(model_matrices, ModelMatrices)
    assert isinstance(model_matrices.model_spec, ModelSpecs)

    # Validate invalid type checking
    with pytest.raises(TypeError, match="`ModelMatrices` instances expect all.*"):
        ModelMatrices("invalid type!")


@pytest.mark.parametrize("ordering", ["degree", "none", "sort"])
def test_model_matrices_preserve_ordering(ordering):
    data = pandas.DataFrame(
        {
            "y": numpy.random.standard_normal(100),
            "x": numpy.random.standard_normal(100),
            "w": numpy.random.standard_normal(100),
            "d": numpy.random.choice(["a", "b", "c"], size=100),
            "e": numpy.random.choice(["a", "b", "c"], size=100),
        }
    )
    formula = Formula("y ~ 1 + e + x + d:x", _ordering=ordering)
    ordering_method = OrderingMethod(ordering)
    assert formula._ordering == ordering_method
    model_matrices = model_matrix(formula, data)
    assert model_matrices.rhs.model_spec.formula._ordering == ordering_method
    assert model_matrices.lhs.model_spec.formula._ordering == ordering_method
