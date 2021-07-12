from collections import OrderedDict

import pytest

import pandas
from formulaic import Formula


class TestModelSpec:
    @pytest.fixture
    def data(self):
        return pandas.DataFrame(
            {
                "A": ["a", "b", "c"],
                "a": [0, 0, 1],
            }
        )

    @pytest.fixture
    def data2(self):
        return pandas.DataFrame(
            {
                "A": ["a", "a", "a"],
                "a": [0, 0, 1],
            }
        )

    @pytest.fixture
    def formula(self):
        return Formula("a + A + a:A")

    @pytest.fixture
    def model_spec(self, formula, data):
        return formula.get_model_matrix(data).model_spec

    def test_attributes(self, model_spec):
        assert model_spec.formula == Formula("a + A + a:A")
        assert model_spec.ensure_full_rank is True
        assert model_spec.materializer == "pandas"
        assert model_spec.feature_names == [
            "Intercept",
            "A[T.b]",
            "A[T.c]",
            "a",
            "A[T.b]:a",
            "A[T.c]:a",
        ]
        assert model_spec.feature_indices == OrderedDict(
            [
                ("Intercept", 0),
                ("A[T.b]", 1),
                ("A[T.c]", 2),
                ("a", 3),
                ("A[T.b]:a", 4),
                ("A[T.c]:a", 5),
            ]
        )
        assert model_spec.term_slices == OrderedDict(
            [
                ("1", slice(0, 1)),
                ("A", slice(1, 3)),
                ("a", slice(3, 4)),
                ("A:a", slice(4, 6)),
            ]
        )

    def test_get_model_matrix(self, model_spec, data2):
        m = model_spec.get_model_matrix(data2)

        assert isinstance(m, pandas.DataFrame)
        assert list(m.columns) == model_spec.feature_names

        model_spec.materializer = None
        m2 = model_spec.get_model_matrix(data2)
        assert isinstance(m2, pandas.DataFrame)
        assert list(m2.columns) == model_spec.feature_names

    def test_differentiate(self, model_spec, formula):
        assert model_spec.differentiate("a").formula == formula.differentiate("a")
