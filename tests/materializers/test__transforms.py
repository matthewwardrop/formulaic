import numpy
import pytest
import scipy.sparse as spsparse

from formulaic.errors import DataMismatchWarning
from formulaic.materializers._transforms import center, encode_categorical


def test_center():
    state = {}
    assert numpy.allclose(center(data=[1, 2, 3], state=state), [-1, 0, 1])
    assert state == {'mean': 2}
    assert numpy.allclose(center(data=[5, 6, 7], state=state), [3, 4, 5])


def test_center_sparse():
    state = {}
    m = spsparse.csc_matrix([1, 2, 3]).transpose()
    assert numpy.allclose(center(data=m, state=state), [-1, 0, 1])


def test_encode_categorical():
    state = {}
    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], state=state),
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
            encode_categorical(data=['a', 'b', 'd', 'a', 'b', 'd'], state=state),
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
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], state=state, reduced_rank=True),
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
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], state=state, reduced_rank=False, spans_intercept=False),
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
    _compare_formulaic_dict(
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], state=state, config={'sparse': True}),
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
        encode_categorical(data=['a', 'b', 'c', 'a', 'b', 'c'], state=state, config={'sparse': True}, reduced_rank=True),
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


def _compare_formulaic_dict(a, b, comp=lambda x, y: numpy.allclose(x, y)):
    assert isinstance(a, dict) and isinstance(b, dict)
    assert len(a) == len(b)
    for key, value in a.items():
        if isinstance(key, str) and key.startswith('__'):
            assert value == b[key]
        else:
            assert comp(value.toarray()[:, 0] if isinstance(value, spsparse.csc_matrix) else value, b[key])
