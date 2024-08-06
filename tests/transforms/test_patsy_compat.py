import numpy
import pandas

from formulaic import model_matrix
from formulaic.transforms.contrasts import TreatmentContrasts
from formulaic.transforms.patsy_compat import Treatment, standardize
from formulaic.transforms.scale import scale


def test_standardize():
    assert numpy.all(
        standardize([1, 2, 3], rescale=False) == scale([1, 2, 3], scale=False)
    )


def test_Treatment():
    assert isinstance(Treatment(reference="a"), TreatmentContrasts)
    assert Treatment(reference="a").base == "a"
    assert numpy.all(
        model_matrix(
            "C(x, Treatment('a'))", pandas.DataFrame({"x": ["a", "b", "c"]})
        ).values
        == numpy.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])
    )


def test_Q():
    assert model_matrix(
        "Q('x')", pandas.DataFrame({"x": [1, 2, 3]})
    ).model_spec.column_names == ("Intercept", "Q('x')")
