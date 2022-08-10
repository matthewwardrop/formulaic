import pandas
import pytest

from formulaic import model_matrix, ModelMatrices, ModelSpecs


def test_model_matrices():
    model_matrices = model_matrix(
        "y ~ x", pandas.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
    )
    assert isinstance(model_matrices, ModelMatrices)
    assert isinstance(model_matrices.model_spec, ModelSpecs)

    # Validate invalid type checking
    with pytest.raises(TypeError, match="`ModelMatrices` instances expect all.*"):
        ModelMatrices("invalid type!")
