from formulaic.utils.ordered_set import OrderedSet


class TestOrderedSet:
    def test_constructor(self):
        assert OrderedSet() == OrderedSet()
        assert len(OrderedSet()) == 0
        assert OrderedSet() != "a"

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
        assert OrderedSet(["z", "a", "b"]) & OrderedSet(["a", "c", "z"]) == OrderedSet(["z", "a"])
        assert ["b"] | OrderedSet("a") == OrderedSet("ba")
        assert ["a", "b"] ^ OrderedSet(["a", "c"]) == OrderedSet(["b", "c"])
        assert ["z", "a", "b"] & OrderedSet(["a", "c", "z"]) == OrderedSet(["z", "a"])
        assert ["z", "a", "b"] - OrderedSet(["a", "c"]) == OrderedSet(["z", "b"])

        s = OrderedSet(["a"])
        s.add("b")
        assert list(s) == ["a", "b"]
        assert s.get_multiplicity("a") == 1
        s.discard("a")
        assert list(s) == ["b"]
        assert s.get_multiplicity("a") == 0
        s.add("b", count=2)
        assert s.get_multiplicity("b") == 3
        s.discard("b", count=1)
        assert s.get_multiplicity("b") == 2
        s.discard("b")
        assert list(s) == []

        s.update(["c", "d"])
        assert list(s) == ["c", "d"]
        s.update({"e": 3})
        assert list(s) == ["c", "d", "e"]
        assert s.get_multiplicity("e") == 3

    def test_sequence_operations(self):
        s = OrderedSet(["z", "k"])
        assert s[0] == "z"
        assert isinstance(s[1:], OrderedSet)
        assert s[1:] == OrderedSet(["k"])
        assert len(s) == 2

        s[0] = "a"
        assert s[0] == "a"
        assert s == OrderedSet(["a", "k"])

        del s[0]
        assert s[0] == "k"
        assert s == OrderedSet(["k"])
        del s[0:]
        assert len(s) == 0

        s.insert(0, "a")
        s.insert(0, "b")
        assert s == ["b", "a"]
        s.insert(0, "a")
        assert s == ["a", "b"]
        assert s.get_multiplicity("a") == 2
        s[10:12] = ["c", "d"]
        assert s == ["a", "b", "c", "d"]
        s[1:3] = ["d", "a"]
        assert s == ["d", "a"]
        assert s.get_multiplicity("a") == 3
        assert s.get_multiplicity("d") == 2