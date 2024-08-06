import pytest

from formulaic.utils.iterators import peekable_iter


def test_peekable_iter():
    it = peekable_iter([1, 2, 3, 4, 5, 6])

    assert it.peek() == 1
    assert it._next == [1]
    assert next(it) == 1
    assert it._next == []
    assert it.peek() == 2

    assert list(it) == [2, 3, 4, 5, 6]
    assert it.peek(None) is None

    with pytest.raises(StopIteration):
        it.peek()
