import numpy
import pytest
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.materializers.transforms import encode_categorical
from formulaic.model_spec import ModelSpec


def test_encode_categorical():
    state = {}
    spec = ModelSpec([], output='pandas')
    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], _state=state, _spec=spec),
        {
            '__kind__': 'categorical',
            '__spans_intercept__': True,
            '__drop_field__': 'a',
            '__format__': '{name}[T.{field}]',
            '__encoded__': True,
            'a': [1, 0, 0, 1, 0, 0],
            'b': [0, 1, 0, 0, 1, 0],
            'c': [0, 0, 1, 0, 0, 1],
        }
    )
    assert state['categories'] == ['a', 'b', 'c']

    with pytest.warns(DataMismatchWarning):
        _compare_formulaic_dict(
            encode_categorical(data=['a', 'b', 'd', 'a', 'b', 'd'], _state=state, _spec=spec),
            {
                '__kind__': 'categorical',
                '__spans_intercept__': True,
                '__drop_field__': 'a',
                '__format__': '{name}[T.{field}]',
                '__encoded__': True,
                'a': [1, 0, 0, 1, 0, 0],
                'b': [0, 1, 0, 0, 1, 0],
                'c': [0, 0, 0, 0, 0, 0],
            }
        )
        assert state['categories'] == ['a', 'b', 'c']

    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], reduced_rank=True, _state=state, _spec=spec),
        {
            '__kind__': 'categorical',
            '__spans_intercept__': False,
            '__drop_field__': None,
            '__format__': '{name}[T.{field}]',
            '__encoded__': True,
            'b': [0, 1, 0, 0, 1, 0],
            'c': [0, 0, 1, 0, 0, 1],
        }
    )
    assert state['categories'] == ['a', 'b', 'c']

    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], reduced_rank=False, spans_intercept=False, _state=state, _spec=spec),
        {
            '__kind__': 'categorical',
            '__spans_intercept__': False,
            '__drop_field__': None,
            '__format__': '{name}[T.{field}]',
            '__encoded__': True,
            'a': [1, 0, 0, 1, 0, 0],
            'b': [0, 1, 0, 0, 1, 0],
            'c': [0, 0, 1, 0, 0, 1],
        }
    )
    assert state['categories'] == ['a', 'b', 'c']


def test_encode_categorical_sparse():
    state = {}
    spec = ModelSpec([], output='sparse')
    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], _state=state, _spec=spec),
        {
            '__kind__': 'categorical',
            '__spans_intercept__': True,
            '__drop_field__': 'a',
            '__format__': '{name}[T.{field}]',
            '__encoded__': True,
            'a': [1, 0, 0, 1, 0, 0],
            'b': [0, 1, 0, 0, 1, 0],
            'c': [0, 0, 1, 0, 0, 1],
        }
    )
    assert state['categories'] == ['a', 'b', 'c']

    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], reduced_rank=True, _state=state, _spec=spec),
        {
            '__kind__': 'categorical',
            '__spans_intercept__': False,
            '__drop_field__': None,
            '__format__': '{name}[T.{field}]',
            '__encoded__': True,
            'b': [0, 1, 0, 0, 1, 0],
            'c': [0, 0, 1, 0, 0, 1],
        }
    )
    assert state['categories'] == ['a', 'b', 'c']

    with pytest.warns(DataMismatchWarning):
        _compare_formulaic_dict(
            encode_categorical(data=['a', 'b', 'd', 'a', 'b', 'd'], _state=state, _spec=spec),
            {
                '__kind__': 'categorical',
                '__spans_intercept__': True,
                '__drop_field__': 'a',
                '__format__': '{name}[T.{field}]',
                '__encoded__': True,
                'a': [1, 0, 0, 1, 0, 0],
                'b': [0, 1, 0, 0, 1, 0],
                'c': [0, 0, 0, 0, 0, 0],
            }
        )
        assert state['categories'] == ['a', 'b', 'c']


def _compare_formulaic_dict(a, b, comp=lambda x, y: numpy.allclose(x, y)):
    assert isinstance(a, dict) and isinstance(b, dict)
    assert sorted(a) == sorted(b)
    for key, value in a.items():
        if isinstance(key, str) and key.startswith('__'):
            assert value == b[key]
        else:
            assert comp(value.toarray()[:, 0] if isinstance(value, spsparse.csc_matrix) else value, b[key])
