import re

import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic import model_matrix
from formulaic.errors import DataMismatchWarning
from formulaic.materializers import FactorValues
from formulaic.model_spec import ModelSpec
from formulaic.transforms.contrasts import (
    ContrastsRegistry as contr,
)
from formulaic.transforms.contrasts import (
    ContrastsState,
    SumContrasts,
    encode_contrasts,
)
from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix


def _compare_factor_values(a, b, comp=lambda x, y: numpy.allclose(x, y)):
    assert type(a) is type(b)
    if isinstance(a, spsparse.csc_matrix):
        assert comp(
            a.toarray(),
            b,
        )
    else:
        assert comp(a, b)
    assert a.__formulaic_metadata__ == b.__formulaic_metadata__


class TestContrastsTransform:
    def test_basic_usage_and_state(self):
        state = {}
        spec = ModelSpec(formula=[], output="pandas")
        _compare_factor_values(
            encode_contrasts(
                data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
                _state=state,
                _spec=spec,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "a": [1, 0, 0, 1, 0, 0],
                        "b": [0, 1, 0, 0, 1, 0],
                        "c": [0, 0, 1, 0, 0, 1],
                    }
                ),
                kind="categorical",
                spans_intercept=True,
                column_names=("a", "b", "c"),
                drop_field="a",
                format="{name}[{field}]",
                format_reduced="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

        with pytest.warns(DataMismatchWarning):
            _compare_factor_values(
                encode_contrasts(
                    data=pandas.Series(["a", "b", "d", "a", "b", "d"]),
                    _state=state,
                    _spec=spec,
                ),
                FactorValues(
                    pandas.DataFrame(
                        {
                            "a": [1, 0, 0, 1, 0, 0],
                            "b": [0, 1, 0, 0, 1, 0],
                            "c": [0, 0, 0, 0, 0, 0],
                        }
                    ),
                    kind="categorical",
                    spans_intercept=True,
                    column_names=("a", "b", "c"),
                    drop_field="a",
                    format="{name}[{field}]",
                    format_reduced="{name}[T.{field}]",
                    encoded=True,
                ),
            )
            assert state["categories"] == ["a", "b", "c"]
            assert "contrasts" in state

        _compare_factor_values(
            encode_contrasts(
                data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
                reduced_rank=True,
                _state=state,
                _spec=spec,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "b": [0, 1, 0, 0, 1, 0],
                        "c": [0, 0, 1, 0, 0, 1],
                    }
                ),
                kind="categorical",
                spans_intercept=False,
                column_names=("b", "c"),
                drop_field=None,
                format="{name}[T.{field}]",
                format_reduced="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

    def test_sparse(self):
        state = {}
        spec = ModelSpec(formula=[], output="sparse")
        _compare_factor_values(
            encode_contrasts(
                data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
                _state=state,
                _spec=spec,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "a": [1, 0, 0, 1, 0, 0],
                        "b": [0, 1, 0, 0, 1, 0],
                        "c": [0, 0, 1, 0, 0, 1],
                    }
                ).values,
                kind="categorical",
                spans_intercept=True,
                column_names=("a", "b", "c"),
                drop_field="a",
                format="{name}[{field}]",
                format_reduced="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

        _compare_factor_values(
            encode_contrasts(
                data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
                reduced_rank=True,
                _state=state,
                _spec=spec,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "b": [0, 1, 0, 0, 1, 0],
                        "c": [0, 0, 1, 0, 0, 1],
                    }
                ).values,
                kind="categorical",
                spans_intercept=False,
                column_names=("b", "c"),
                drop_field=None,
                format="{name}[T.{field}]",
                format_reduced="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

        with pytest.warns(DataMismatchWarning):
            _compare_factor_values(
                encode_contrasts(
                    data=pandas.Series(["a", "b", "d", "a", "b", "d"]),
                    _state=state,
                    _spec=spec,
                ),
                FactorValues(
                    pandas.DataFrame(
                        {
                            "a": [1, 0, 0, 1, 0, 0],
                            "b": [0, 1, 0, 0, 1, 0],
                            "c": [0, 0, 0, 0, 0, 0],
                        }
                    ).values,
                    kind="categorical",
                    spans_intercept=True,
                    column_names=("a", "b", "c"),
                    drop_field="a",
                    format="{name}[{field}]",
                    format_reduced="{name}[T.{field}]",
                    encoded=True,
                ),
            )
            assert state["categories"] == ["a", "b", "c"]
            assert "contrasts" in state

    def test_numpy(self):
        assert isinstance(
            encode_contrasts(
                data=pandas.Series(["a", "b", "c", "a", "b", "c"]), output="numpy"
            ),
            numpy.ndarray,
        )

    def test_specifying_encode_contrasts(self):
        state = {}
        _compare_factor_values(
            encode_contrasts(
                data=["a", "b", "c", "a", "b", "c"],
                contrasts=contr.treatment("c"),
                _state=state,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "a": [1, 0, 0, 1, 0, 0],
                        "b": [0, 1, 0, 0, 1, 0],
                        "c": [0, 0, 1, 0, 0, 1],
                    }
                ),
                kind="categorical",
                spans_intercept=True,
                column_names=("a", "b", "c"),
                drop_field="c",
                format="{name}[{field}]",
                format_reduced="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

    def test_specifying_contrast_class(self):
        state = {}
        _compare_factor_values(
            encode_contrasts(
                data=["a", "b", "c", "a", "b", "c"],
                contrasts=contr.treatment,
                _state=state,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "a": [1, 0, 0, 1, 0, 0],
                        "b": [0, 1, 0, 0, 1, 0],
                        "c": [0, 0, 1, 0, 0, 1],
                    }
                ),
                kind="categorical",
                spans_intercept=True,
                column_names=("a", "b", "c"),
                drop_field="a",
                format="{name}[{field}]",
                format_reduced="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

    def test_specifying_custom_encode_contrasts(self):
        state = {}
        _compare_factor_values(
            encode_contrasts(
                data=["a", "b", "c", "a", "b", "c"],
                contrasts={"ordinal": [1, 2, 3]},
                _state=state,
            ),
            FactorValues(
                pandas.DataFrame(
                    {
                        "ordinal": [1, 2, 3, 1, 2, 3],
                    }
                ),
                kind="categorical",
                spans_intercept=False,
                column_names=("ordinal",),
                drop_field=None,
                format="{name}[{field}]",
                format_reduced="{name}[{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]
        assert "contrasts" in state

    def test_invalid_output_type(self):
        with pytest.raises(ValueError, match=r"^Unknown output type"):
            encode_contrasts(data=["a", "b", "c", "a", "b", "c"], output="invalid")

    @pytest.mark.filterwarnings("ignore::formulaic.errors.DataMismatchWarning")
    def test_empty_levels(self):
        empty_numpy = encode_contrasts(
            data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
            levels=[],
            output="numpy",
        )
        assert empty_numpy.shape == (6, 0)

        empty_numpy_reduced = encode_contrasts(
            data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
            levels=["a"],
            output="numpy",
            reduced_rank=True,
        )
        assert empty_numpy_reduced.shape == (6, 0)

        empty_pandas = encode_contrasts(
            data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
            levels=[],
            output="pandas",
        )
        assert empty_pandas.shape == (6, 0)

        empty_sparse = encode_contrasts(
            data=pandas.Series(["a", "b", "c", "a", "b", "c"]),
            levels=[],
            output="sparse",
        )
        assert empty_sparse.shape == (6, 0)


# Test specific contrasts


@pytest.fixture
def categories():
    return ["a", "b", "c", "a", "b", "c"]


@pytest.fixture
def category_dummies(categories):
    return pandas.get_dummies(categories)


@pytest.fixture
def category_dummies_sparse(categories):
    levels, dummies = categorical_encode_series_to_sparse_csc_matrix(categories)
    return dummies, levels


class TestTreatmentContrasts:
    def test_dense(self, category_dummies):
        contrasts = contr.treatment()

        encoded = contrasts.apply(category_dummies, ["a", "b", "c"])
        assert list(encoded.columns) == ["b", "c"]
        assert encoded.__formulaic_metadata__.drop_field is None
        assert encoded.__formulaic_metadata__.format == "{name}[T.{field}]"
        assert encoded.to_dict("list") == {
            "b": [0, 1, 0, 0, 1, 0],
            "c": [0, 0, 1, 0, 0, 1],
        }

        encoded_spanning = contrasts.apply(
            category_dummies, ["a", "b", "c"], reduced_rank=False
        )
        assert list(encoded_spanning.columns) == ["a", "b", "c"]
        assert encoded_spanning.__formulaic_metadata__.drop_field == "a"
        assert encoded_spanning.__formulaic_metadata__.format == "{name}[{field}]"
        assert encoded_spanning.to_dict("list") == {
            "a": [1, 0, 0, 1, 0, 0],
            "b": [0, 1, 0, 0, 1, 0],
            "c": [0, 0, 1, 0, 0, 1],
        }

        encoded_spanning = contrasts.apply(
            category_dummies.values, ["a", "b", "c"], reduced_rank=False
        )
        assert encoded_spanning.__formulaic_metadata__.drop_field == "a"
        assert numpy.allclose(
            encoded_spanning,
            numpy.array(
                [
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1],
                ]
            ).T,
        )

        encoded_base = contr.treatment("b").apply(category_dummies, ["a", "b", "c"])
        assert list(encoded_base.columns) == ["a", "c"]
        assert encoded_base.__formulaic_metadata__.drop_field is None
        assert encoded_base.to_dict("list") == {
            "a": [1, 0, 0, 1, 0, 0],
            "c": [0, 0, 1, 0, 0, 1],
        }

        with pytest.raises(
            ValueError,
            match=r"Value `'invalid'` for `TreatmentContrasts.base` is not among the provided levels.",
        ):
            contr.treatment("invalid").apply(category_dummies, ["a", "b", "c"])

    def test_sparse(self, category_dummies_sparse):
        contrasts = contr.treatment()

        encoded = contrasts.apply(*category_dummies_sparse)
        assert encoded.__formulaic_metadata__.column_names == ("b", "c")
        assert encoded.__formulaic_metadata__.drop_field is None
        assert numpy.all(
            encoded.toarray()
            == numpy.array(
                [
                    [0, 1, 0, 0, 1, 0],  # b
                    [0, 0, 1, 0, 0, 1],  # c
                ]
            ).T
        )

        encoded_spanning = contrasts.apply(*category_dummies_sparse, reduced_rank=False)
        assert encoded_spanning.__formulaic_metadata__.column_names == ("a", "b", "c")
        assert encoded_spanning.__formulaic_metadata__.drop_field == "a"
        assert numpy.all(
            encoded_spanning.toarray()
            == numpy.array(
                [
                    [1, 0, 0, 1, 0, 0],  # a
                    [0, 1, 0, 0, 1, 0],  # b
                    [0, 0, 1, 0, 0, 1],  # c
                ]
            ).T
        )

        encoded_base = contr.treatment("b").apply(*category_dummies_sparse)
        assert encoded_base.__formulaic_metadata__.column_names == ("a", "c")
        assert encoded_base.__formulaic_metadata__.drop_field is None
        assert numpy.all(
            encoded_base.toarray()
            == numpy.array(
                [
                    [1, 0, 0, 1, 0, 0],  # a
                    [0, 0, 1, 0, 0, 1],  # c
                ]
            ).T
        )

        assert numpy.allclose(
            contrasts.get_coefficient_matrix(
                levels=["a", "b", "c"], reduced_rank=False, sparse=True
            ).toarray(),
            numpy.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
        )

    def test_coding_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix = contr.treatment().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coding_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_reduced = contr.treatment().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.all(coding_matrix_reduced == reference_reduced)

        coding_matrix_reduced_sparse = contr.treatment().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True, sparse=True
        )
        assert numpy.all(
            coding_matrix_reduced_sparse.toarray() == reference_reduced.values
        )

    def test_coefficient_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coefficient_matrix = contr.treatment().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coefficient_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "a": [1, -1, -1],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b-a", "c-a"],
        )
        coefficient_matrix_reduced = contr.treatment().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.all(coefficient_matrix_reduced == reference_reduced)


class TestSASContrasts:
    # Mostly covered by `TreatmentContrasts`, we'll just look at the delta.

    def test_basic(self, category_dummies):
        encoded = contr.SAS().apply(category_dummies, ["a", "b", "c"])
        assert list(encoded.columns) == ["a", "b"]
        assert encoded.__formulaic_metadata__.drop_field is None
        assert encoded.to_dict("list") == {
            "a": [1, 0, 0, 1, 0, 0],
            "b": [0, 1, 0, 0, 1, 0],
        }

        encoded = contr.SAS("b").apply(
            category_dummies, ["a", "b", "c"], reduced_rank=False
        )
        assert list(encoded.columns) == ["a", "b", "c"]
        assert encoded.__formulaic_metadata__.drop_field == "b"
        assert encoded.to_dict("list") == {
            "a": [1, 0, 0, 1, 0, 0],
            "b": [0, 1, 0, 0, 1, 0],
            "c": [0, 0, 1, 0, 0, 1],
        }

        with pytest.raises(
            ValueError,
            match=r"Value `'invalid'` for `SASContrasts.base` is not among the provided levels.",
        ):
            contr.SAS("invalid").apply(category_dummies, ["a", "b", "c"])


class TestSumContrasts:
    def test_coding_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix = contr.sum().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coding_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "a": [1, 0, -1],
                "b": [0, 1, -1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_reduced = contr.sum().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.all(coding_matrix_reduced == reference_reduced)

        coding_matrix_reduced_sparse = contr.sum().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True, sparse=True
        )
        assert numpy.all(
            coding_matrix_reduced_sparse.toarray() == reference_reduced.values
        )

    def test_coefficient_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coefficient_matrix = contr.sum().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coefficient_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "a": [1.0 / 3, 2.0 / 3, -1.0 / 3],
                "b": [1.0 / 3, -1.0 / 3, 2.0 / 3],
                "c": [1.0 / 3, -1.0 / 3, -1.0 / 3],
            },
            index=["avg", "a - avg", "b - avg"],
        )
        coefficient_matrix_reduced = contr.sum().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coefficient_matrix_reduced, reference_reduced)

    def test_get_drop_field(self):
        # test once here for all constrasts that don't override this
        assert SumContrasts().get_drop_field(["a", "b", "c"]) is None
        assert SumContrasts().get_drop_field(["a", "b", "c"], reduced_rank=False) == "a"

    def test_get_factor_format(self):
        assert (
            SumContrasts().get_factor_format(None, reduced_rank=False)
            == "{name}[{field}]"
        )
        assert (
            SumContrasts().get_factor_format(None, reduced_rank=True)
            == "{name}[S.{field}]"
        )


class TestHelmertContrasts:
    def test_coding_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix = contr.helmert().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coding_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "b": [-1, 1, 0],
                "c": [-1, -1, 2],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_reduced = contr.helmert().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.all(coding_matrix_reduced == reference_reduced)

        coding_matrix_reduced_sparse = contr.helmert().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True, sparse=True
        )
        assert numpy.all(
            coding_matrix_reduced_sparse.toarray() == reference_reduced.values
        )

        reference_forward = pandas.DataFrame(
            {
                "a": [2, -1, -1],
                "b": [0, 1, -1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_forward = contr.helmert(reverse=False).get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.all(coding_matrix_forward == reference_forward)

        reference_scaled = pandas.DataFrame(
            {
                "b": [-0.5, 0.5, 0],
                "c": [-1 / 3, -1 / 3, 2 / 3],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_scaled = contr.helmert(scale=True).get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.all(coding_matrix_scaled == reference_scaled)

    def test_coefficient_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coefficient_matrix = contr.helmert().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coefficient_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "a": [1.0 / 3, -1.0 / 2, -1.0 / 6],
                "b": [1.0 / 3, 1.0 / 2, -1.0 / 6],
                "c": [1.0 / 3, 0, 1.0 / 3],
            },
            index=["avg", "b - rolling_avg", "c - rolling_avg"],
        )
        coefficient_matrix_reduced = contr.helmert().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coefficient_matrix_reduced, reference_reduced)

    def test_get_factor_format(self):
        assert (
            contr.helmert().get_factor_format(None, reduced_rank=False)
            == "{name}[{field}]"
        )
        assert (
            contr.helmert().get_factor_format(None, reduced_rank=True)
            == "{name}[H.{field}]"
        )


class TestDiffContrasts:
    def test_coding_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix = contr.diff().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coding_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "b": [-2.0 / 3, 1.0 / 3, 1.0 / 3],
                "c": [-1.0 / 3, -1.0 / 3, 2.0 / 3],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_reduced = contr.diff().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coding_matrix_reduced, reference_reduced)

        coding_matrix_reduced_sparse = contr.diff().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True, sparse=True
        )
        assert numpy.allclose(
            coding_matrix_reduced_sparse.toarray(), reference_reduced.values
        )

        reference_forward = pandas.DataFrame(
            {
                "a": [2.0 / 3, -1.0 / 3, -1.0 / 3],
                "b": [1.0 / 3, 1.0 / 3, -2.0 / 3],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_forward = contr.diff(backward=False).get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coding_matrix_forward, reference_forward)

    def test_coefficient_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coefficient_matrix = contr.diff().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coefficient_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "a": [1.0 / 3, -1, 0],
                "b": [1.0 / 3, 1, -1],
                "c": [1.0 / 3, 0, 1],
            },
            index=["avg", "b - a", "c - b"],
        )
        coefficient_matrix_reduced = contr.diff().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coefficient_matrix_reduced, reference_reduced)

    def test_get_factor_format(self):
        assert (
            contr.diff().get_factor_format(None, reduced_rank=False)
            == "{name}[{field}]"
        )
        assert (
            contr.diff().get_factor_format(None, reduced_rank=True)
            == "{name}[D.{field}]"
        )


class TestPolyContrasts:
    def test_coding_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coding_matrix = contr.poly().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coding_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                ".Q": [-0.7071067811865475, 0.0, 0.7071067811865475],
                ".L": [0.4082482904638631, -0.816496580927726, 0.4082482904638631],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_reduced = contr.poly().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coding_matrix_reduced, reference_reduced)

        coding_matrix_reduced_sparse = contr.poly().get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True, sparse=True
        )
        assert numpy.allclose(
            coding_matrix_reduced_sparse.toarray(), reference_reduced.values
        )

        reference_scores = pandas.DataFrame(
            {
                ".L": [-0.47075654176200415, -0.34236839400873026, 0.8131249357707346],
                ".Q": [0.6671243849949887, -0.7412493166611032, 0.07412493166611252],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_scores = contr.poly(scores=[10, 11, 20]).get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coding_matrix_scores, reference_scores)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "`PolyContrasts.scores` must have the same cardinality as the categories."
            ),
        ):
            contr.poly(scores=[1, 2, 3]).get_coding_matrix(levels=["a", "b"])

    def test_coefficient_matrix(self):
        reference = pandas.DataFrame(
            {
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1],
            },
            index=["a", "b", "c"],
        )
        coefficient_matrix = contr.poly().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coefficient_matrix == reference)

        reference_reduced = pandas.DataFrame(
            {
                "avg": [1.0 / 3, 1.0 / 3, 1.0 / 3],
                ".Q": [-0.7071067811865475, 0.0, 0.7071067811865475],
                ".L": [0.4082482904638631, -0.816496580927726, 0.4082482904638631],
            },
            index=["a", "b", "c"],
        ).T
        coefficient_matrix_reduced = contr.poly().get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coefficient_matrix_reduced, reference_reduced)

    def test_get_factor_format(self):
        assert (
            contr.poly().get_factor_format(None, reduced_rank=False)
            == "{name}[{field}]"
        )
        assert (
            contr.poly().get_factor_format(None, reduced_rank=True) == "{name}[{field}]"
        )


class TestCustomContrasts:
    def test_coding_matrix(self):
        contrasts = contr.custom({"ordinal": [1, 2, 3]})

        reference = pandas.DataFrame(
            {
                "ordinal": [1, 2, 3],
            },
            index=["a", "b", "c"],
        )
        coding_matrix = contrasts.get_coding_matrix(["a", "b", "c"], reduced_rank=False)
        assert numpy.all(coding_matrix == reference)

        coding_matrix_reduced = contrasts.get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coding_matrix_reduced, reference)

        coding_matrix_reduced_sparse = contrasts.get_coding_matrix(
            ["a", "b", "c"], reduced_rank=True, sparse=True
        )
        assert numpy.allclose(coding_matrix_reduced_sparse.toarray(), reference.values)

        # Do the same thing for numpy arrays
        contrasts = contr.custom([[1], [2], [3]], names=["ordinal"])
        coding_matrix = contrasts.get_coding_matrix(["a", "b", "c"], reduced_rank=False)
        assert numpy.all(coding_matrix == reference)

        # And again without names
        contrasts_unnamed = contr.custom([[1], [2], [3]])
        reference_unnamed = pandas.DataFrame(
            {
                1: [1, 2, 3],
            },
            index=["a", "b", "c"],
        )
        coding_matrix_unnamed = contrasts_unnamed.get_coding_matrix(
            ["a", "b", "c"], reduced_rank=False
        )
        assert numpy.all(coding_matrix_unnamed == reference_unnamed)

        with pytest.raises(
            ValueError,
            match=r"Names must be aligned with the columns of the contrast array\.",
        ):
            contrasts = contr.custom([[1], [2], [3]], names=["ordinal", "invalid"])

    def test_coefficient_matrix(self):
        contrasts = contr.custom({"x": [1, -1, 2], "y": [-1, 1, 2]})

        reference_reduced = pandas.DataFrame(
            {
                "a": [0.5, 0.125, -0.375],
                "b": [0.5, -0.375, 0.125],
                "c": [0.0, 0.25, 0.25],
            },
            index=[1, 2, 3],
        )
        coefficient_matrix_reduced = contrasts.get_coefficient_matrix(
            ["a", "b", "c"], reduced_rank=True
        )
        assert numpy.allclose(coefficient_matrix_reduced, reference_reduced)


def test_full_rankness_opt_out():
    data = pandas.DataFrame({"A": ["a", "b", "c"]})
    assert model_matrix("A", data).model_spec.column_names == (
        "Intercept",
        "A[T.b]",
        "A[T.c]",
    )
    assert model_matrix("C(A)", data).model_spec.column_names == (
        "Intercept",
        "C(A)[T.b]",
        "C(A)[T.c]",
    )
    assert model_matrix(
        "C(A, spans_intercept=False)", data
    ).model_spec.column_names == (
        "Intercept",
        "C(A, spans_intercept=False)[a]",
        "C(A, spans_intercept=False)[b]",
        "C(A, spans_intercept=False)[c]",
    )


def test_contrasts_state():
    assert numpy.allclose(
        ContrastsState(contr.helmert(), ["a", "b", "c"]).get_coding_matrix(),
        contr.helmert().get_coding_matrix(["a", "b", "c"]),
    )
    assert numpy.allclose(
        ContrastsState(contr.helmert(), ["a", "b", "c"]).get_coefficient_matrix(),
        contr.helmert().get_coefficient_matrix(["a", "b", "c"]),
    )
