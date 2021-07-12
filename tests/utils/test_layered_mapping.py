import pytest

from formulaic.utils.layered_mapping import LayeredMapping


def test_layered_context():

    layer1 = {"a": 1, "b": 2, "c": 3}
    layer2 = {"a": 2, "d": 4}

    layered = LayeredMapping(layer1, layer2)

    assert layered["a"] == 1
    assert layered["b"] == 2
    assert layered["c"] == 3
    assert layered["d"] == 4

    with pytest.raises(KeyError):
        layered["e"]

    assert len(layered) == 4
    assert set(layered) == {"a", "b", "c", "d"}

    layered2 = layered.with_layers({"e": 0, "f": 1, "g": 2}, {"h": 1})
    assert set(layered2) == {"a", "b", "c", "d", "e", "f", "g", "h"}

    layered.with_layers({"e": 2}, inplace=True)
    assert set(layered) == {"a", "b", "c", "d", "e"}

    assert layered.with_layers() is layered

    # Test mutations
    layered["f"] = 10
    assert layered.mutations == {"f": 10}

    del layered["f"]
    assert layered.mutations == {}

    with pytest.raises(KeyError):
        del layered["a"]
