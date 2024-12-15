from formulaic.parser.types import OrderedSet


class TestOrderedSet:
    def test_constructor(self):
        assert OrderedSet() == OrderedSet()
        assert len(OrderedSet()) == 0

    def test_multiplicity(self):
        assert list(OrderedSet(["a", "a", "z", "b"])) == ["a", "z", "b"]
        assert OrderedSet(["a", "a", "z", "b"]).get_multiplicity("a") == 2
        assert OrderedSet(["a", "a", "z", "b"]).get_multiplicity("b") == 1
        assert OrderedSet(["a", "a", "z", "b"]).get_multiplicity("missing") == 0

    def test_repr(self):
        assert repr(OrderedSet(["z", "b", "c"])) == "{'z', 'b', 'c'}"

    def test_set_operations(self):
        assert OrderedSet(["z", "k"]) | ["a", "b"] == OrderedSet(["z", "k", "a", "b"])
        assert OrderedSet(("z", "k")) - ("z",) == OrderedSet("k")
        assert ["b"] | OrderedSet("a") == OrderedSet("ba")

        s = OrderedSet(["a"])
        s.add("b")
        assert list(s) == ["a", "b"]
        assert s.get_multiplicity("a") == 1
        s.discard("a")
        assert list(s) == ["b"]
        assert s.get_multiplicity("a") == 0

        s.update(["c", "d"])
        assert list(s) == ["b", "c", "d"]

    def test_sequence_operations(self):
        s = OrderedSet(["z", "k"])
        assert s[0] == "z"
        assert isinstance(s[1:], OrderedSet)
        assert s[1:] == OrderedSet(["k"])
        assert len(s) == 2
