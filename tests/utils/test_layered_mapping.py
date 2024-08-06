import re

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
    assert layered._mutations == {"f": 10}

    del layered["f"]
    assert layered._mutations == {}

    with pytest.raises(KeyError):
        del layered["a"]


def test_named_layered_mappings():
    data_layer = LayeredMapping({"data": 1}, name="data")
    context_layer = LayeredMapping({"context": "context"}, name="context")
    layers = LayeredMapping({"data": None, "context": None}, data_layer, context_layer)

    assert sorted(layers.named_layers) == ["context", "data"]
    assert layers["data"] is None
    assert layers["context"] is None
    assert layers.data["data"] == 1
    assert layers.context["context"] == "context"

    assert layers.with_layers({"data": 2}, inplace=True)["data"] == 2
    assert sorted(
        layers.with_layers({"data": 2}, inplace=True, name="toplevel").named_layers
    ) == ["context", "data", "toplevel"]

    with pytest.raises(
        AttributeError,
        match=re.escape("'missing' does not correspond to a named layer."),
    ):
        layers.missing

    full_layers = LayeredMapping(data_layer, context_layer, name="toplevel")
    full_layers["local"] = True
    assert full_layers.get_with_layer_name("local") == (True, "toplevel")
    assert full_layers.get_with_layer_name("data") == (1, "toplevel:data")
    assert full_layers.get_with_layer_name("missing") == (None, None)
    assert full_layers.get_layer_name_for_key("data") == "toplevel:data"
