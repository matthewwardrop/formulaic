import copy
import pickle

import numpy
import pandas
import pytest

from formulaic import (
    FactorValues,
    ModelMatrices,
    ModelMatrix,
    ModelSpec,
    ModelSpecs,
    model_matrix,
)


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
