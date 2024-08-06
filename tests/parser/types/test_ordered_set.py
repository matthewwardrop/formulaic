from formulaic.parser.types import OrderedSet


def test_ordered_set():
    assert OrderedSet() == OrderedSet()
    assert len(OrderedSet()) == 0

    assert list(OrderedSet(["a", "a", "z", "b"])) == ["a", "z", "b"]
    assert repr(OrderedSet(["z", "b", "c"])) == "{'z', 'b', 'c'}"

    assert OrderedSet(["z", "k"]) | ["a", "b"] == OrderedSet(["z", "k", "a", "b"])
    assert OrderedSet(("z", "k")) - ("z",) == OrderedSet(("k"))
    assert ["b"] | OrderedSet("a") == OrderedSet("ba")
