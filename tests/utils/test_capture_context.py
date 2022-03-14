import pytest

from formulaic.utils.context import capture_context


def test_capture_context():
    A = 1

    def nested_context(context=0):
        A = 2
        return capture_context(context)

    assert capture_context()["A"] == 1
    assert nested_context()["A"] == 2
    assert nested_context(1)["A"] == 1

    with pytest.raises(ValueError, match="call stack is not deep enough"):
        assert capture_context(1000)
