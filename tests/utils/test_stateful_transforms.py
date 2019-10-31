from formulaic.utils.stateful_transforms import stateful_eval, stateful_transform


@stateful_transform
def dummy_transform(data, state, config):
    assert config.sparse is False
    if 'data' not in state:
        state['data'] = data
    return state['data']


def test_stateful_transform():

    state = {}
    assert dummy_transform(1, state=state) == 1
    assert state['data'] == 1
    assert dummy_transform(2, state=state) == 1
    assert dummy_transform(2) == 2


def test_stateful_eval():
    state = {}

    assert stateful_eval("dummy_transform(data)", {'dummy_transform': dummy_transform, 'data': 1}, state, None) == 1
    assert state == {"dummy_transform(data)": {'data': 1}}
    assert stateful_eval("dummy_transform(data)", {'dummy_transform': dummy_transform, 'data': 2}, state, None) == 1


def test_stateful_eval_distribution():
    state = {}

    assert stateful_eval("dummy_transform(data)", {'dummy_transform': dummy_transform, 'data': {'__property__': 'value', 'a': 1, 'b': 2}}, state, None) == {'__property__': 'value', 'a': 1, 'b': 2}
    assert state == {"dummy_transform(data)": {'a': {'data': 1}, 'b': {'data': 2}}}
    assert stateful_eval("dummy_transform(data)", {'dummy_transform': dummy_transform, 'data': {'__property__': 'value2', 'a': 3, 'b': 4}}, state, None) == {'__property__': 'value2', 'a': 1, 'b': 2}
