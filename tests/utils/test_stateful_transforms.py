import pytest

import numpy

from formulaic.utils.stateful_transforms import stateful_eval, stateful_transform


@stateful_transform
def dummy_transform(data, _state=None, _spec=None, _metadata=None):
    if _metadata is not None:
        _metadata["added"] = True
    if "data" not in _state:
        _state["data"] = data
    return _state["data"]


def test_stateful_transform():

    state = {}
    metadata = {}
    assert dummy_transform(1, _state=state, _metadata=metadata) == 1
    assert state["data"] == 1
    assert metadata.get("added") is True
    assert dummy_transform(2, _state=state) == 1
    assert dummy_transform(2) == 2


def test_stateful_eval():
    state = {}

    assert (
        stateful_eval(
            "dummy_transform(data)",
            {"dummy_transform": dummy_transform, "data": 1},
            None,
            state,
            None,
        )
        == 1
    )
    assert state == {"dummy_transform(data)": {"data": 1}}
    assert (
        stateful_eval(
            "dummy_transform(data)",
            {"dummy_transform": dummy_transform, "data": 2},
            None,
            state,
            None,
        )
        == 1
    )


def test_stateful_eval_variable_name_sanitization():
    assert (
        stateful_eval(
            "`data|a` / `data|b`",
            {"data|a": 1, "data|b": 2, "data_b": 3},
            None,
            None,
            None,
        )
        == 0.5
    )
    assert (
        stateful_eval(
            "`2data|a` / `2data|b`",
            {"2data|a": 1, "2data|b": 2, "2data_b": 3},
            None,
            None,
            None,
        )
        == 0.5
    )


def test_stateful_eval_distribution():
    state = {}

    assert stateful_eval(
        "dummy_transform(data)",
        {
            "dummy_transform": dummy_transform,
            "data": {"__property__": "value", "a": 1, "b": 2},
        },
        None,
        state,
        None,
    ) == {"__property__": "value", "a": 1, "b": 2}
    assert state == {"dummy_transform(data)": {"a": {"data": 1}, "b": {"data": 2}}}
    assert stateful_eval(
        "dummy_transform(data)",
        {
            "dummy_transform": dummy_transform,
            "data": {"__property__": "value2", "a": 3, "b": 4},
        },
        None,
        state,
        None,
    ) == {"__property__": "value2", "a": 1, "b": 2}


def test_stateful_eval_func_attr():
    assert stateful_eval("numpy.exp(0)", {"numpy": numpy}, None, {}, None) == 1.0

    with pytest.raises(NameError):
        assert stateful_eval("non_existent.me_too(0)", {}, None, {}, None)
