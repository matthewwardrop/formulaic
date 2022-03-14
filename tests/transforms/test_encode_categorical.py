import numpy
import pytest
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.materializers import FactorValues
from formulaic.model_spec import ModelSpec
from formulaic.transforms import encode_categorical


def test_encode_categorical():
    state = {}
    spec = ModelSpec([], output="pandas")
    _compare_factor_values(
        encode_categorical(
            data=["a", "b", "c", "a", "b", "c"], _state=state, _spec=spec
        ),
        FactorValues(
            {
                "a": [1, 0, 0, 1, 0, 0],
                "b": [0, 1, 0, 0, 1, 0],
                "c": [0, 0, 1, 0, 0, 1],
            },
            kind="categorical",
            spans_intercept=True,
            drop_field="a",
            format="{name}[T.{field}]",
            encoded=True,
        ),
    )
    assert state["categories"] == ["a", "b", "c"]

    with pytest.warns(DataMismatchWarning):
        _compare_factor_values(
            encode_categorical(
                data=["a", "b", "d", "a", "b", "d"], _state=state, _spec=spec
            ),
            FactorValues(
                {
                    "a": [1, 0, 0, 1, 0, 0],
                    "b": [0, 1, 0, 0, 1, 0],
                    "c": [0, 0, 0, 0, 0, 0],
                },
                kind="categorical",
                spans_intercept=True,
                drop_field="a",
                format="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]

    _compare_factor_values(
        encode_categorical(
            data=["a", "b", "c", "a", "b", "c"],
            reduced_rank=True,
            _state=state,
            _spec=spec,
        ),
        FactorValues(
            {
                "b": [0, 1, 0, 0, 1, 0],
                "c": [0, 0, 1, 0, 0, 1],
            },
            kind="categorical",
            spans_intercept=False,
            drop_field=None,
            format="{name}[T.{field}]",
            encoded=True,
        ),
    )
    assert state["categories"] == ["a", "b", "c"]

    _compare_factor_values(
        encode_categorical(
            data=["a", "b", "c", "a", "b", "c"],
            reduced_rank=False,
            spans_intercept=False,
            _state=state,
            _spec=spec,
        ),
        FactorValues(
            {
                "a": [1, 0, 0, 1, 0, 0],
                "b": [0, 1, 0, 0, 1, 0],
                "c": [0, 0, 1, 0, 0, 1],
            },
            kind="categorical",
            spans_intercept=False,
            drop_field=None,
            format="{name}[T.{field}]",
            encoded=True,
        ),
    )
    assert state["categories"] == ["a", "b", "c"]


def test_encode_categorical_sparse():
    state = {}
    spec = ModelSpec([], output="sparse")
    _compare_factor_values(
        encode_categorical(
            data=["a", "b", "c", "a", "b", "c"], _state=state, _spec=spec
        ),
        FactorValues(
            {
                "a": [1, 0, 0, 1, 0, 0],
                "b": [0, 1, 0, 0, 1, 0],
                "c": [0, 0, 1, 0, 0, 1],
            },
            kind="categorical",
            spans_intercept=True,
            drop_field="a",
            format="{name}[T.{field}]",
            encoded=True,
        ),
    )
    assert state["categories"] == ["a", "b", "c"]

    _compare_factor_values(
        encode_categorical(
            data=["a", "b", "c", "a", "b", "c"],
            reduced_rank=True,
            _state=state,
            _spec=spec,
        ),
        FactorValues(
            {
                "b": [0, 1, 0, 0, 1, 0],
                "c": [0, 0, 1, 0, 0, 1],
            },
            kind="categorical",
            spans_intercept=False,
            drop_field=None,
            format="{name}[T.{field}]",
            encoded=True,
        ),
    )
    assert state["categories"] == ["a", "b", "c"]

    with pytest.warns(DataMismatchWarning):
        _compare_factor_values(
            encode_categorical(
                data=["a", "b", "d", "a", "b", "d"], _state=state, _spec=spec
            ),
            FactorValues(
                {
                    "a": [1, 0, 0, 1, 0, 0],
                    "b": [0, 1, 0, 0, 1, 0],
                    "c": [0, 0, 0, 0, 0, 0],
                },
                kind="categorical",
                spans_intercept=True,
                drop_field="a",
                format="{name}[T.{field}]",
                encoded=True,
            ),
        )
        assert state["categories"] == ["a", "b", "c"]


def _compare_factor_values(a, b, comp=lambda x, y: numpy.allclose(x, y)):
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert sorted(a) == sorted(b)
        for key, value in a.items():
            assert comp(
                value.toarray()[:, 0]
                if isinstance(value, spsparse.csc_matrix)
                else value,
                b[key],
            )
    else:
        assert comp(a, b)
    assert a.__formulaic_metadata__ == b.__formulaic_metadata__
