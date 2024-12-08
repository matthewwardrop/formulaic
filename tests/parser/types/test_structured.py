import pickle
import re
from io import BytesIO

import pytest

from formulaic.utils.structured import Structured


class TestStructured:
    def test_constructor(self):
        obj = object()
        assert Structured(obj).root is obj
        assert Structured(obj, key="asd").root is obj
        assert len(Structured(obj, key="asd")) == 2
        assert {
            d for d in dir(Structured(obj, key="asd")) if not d.startswith("_")
        } == {"root", "key"}

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Substructure keys cannot start with an underscore. The invalid keys are: {'_invalid'}."
            ),
        ):
            Structured(_invalid=True)

    def test_access_structure(self):
        s = Structured("Hello", key="asd")
        assert s.root == "Hello"
        assert s[None] == "Hello"
        assert s.key == "asd"
        assert s["key"] == "asd"

        assert Structured(("1", "2"))[0] == "1"

        assert Structured()._has_root is False
        with pytest.raises(AttributeError, match="root"):
            Structured().root
        with pytest.raises(KeyError, match="root"):
            Structured()["root"]

        s2 = Structured({"my_key": "my_value"})
        assert list(s2) == ["my_key"]
        assert s2["my_key"] == "my_value"

        s3 = Structured(["a", "b"])
        assert list(s3) == ["a", "b"]
        assert s3[0] == "a"

        s4 = Structured((1, 2))
        assert s4[0] == 1

        s5 = Structured(Structured((Structured("Hello"),)))
        assert s5[("root", "root", 0, "root")] == "Hello"
        with pytest.raises(KeyError, match="extends beyond structure"):
            s5[("root", "root", 0, "root", 0)]

    def test_item_preparation(self):
        class SubStructured(Structured):
            def _prepare_item(self, key, item):
                return str(item) + "_prepared"

        assert SubStructured("Hello")._to_dict() == {"root": "Hello_prepared"}
        assert SubStructured(SubStructured("Hello"))._to_dict() == {
            "root": {"root": "Hello_prepared"}
        }
        assert SubStructured(Structured("Hello"))._to_dict() == {
            "root": {"root": "Hello_prepared_prepared"}
        }

    def test__map(self):
        assert Structured("Hi", a="Hello", b="Greetings")._map(len)._to_dict() == {
            "root": 2,
            "a": 5,
            "b": 9,
        }
        assert Structured(("Hi", "Dave"), a=["Response", "not", "forthcoming"])._map(
            len
        )._to_dict() == {
            "root": (2, 4),
            "a": 3,
        }
        assert Structured(("Hi", Structured("Hi!")), a=Structured(hello="world"))._map(
            len
        )._to_dict() == {
            "root": (2, {"root": 3}),
            "a": {
                "hello": 5,
            },
        }
        assert Structured((Structured("Hi", a="Hi"),))._map(len)._to_dict() == {
            "root": ({"root": 2, "a": 2},)
        }

        class MyStructured(Structured):
            pass

        assert isinstance(
            Structured("Hi", a="Hey")._map(
                lambda s: f"{s} there!", as_type=MyStructured
            ),
            MyStructured,
        )

    def test__flatten(self):
        assert set(
            Structured("Hi", a="Hello", b=Structured(c="Greetings"))._flatten()
        ) == {"Hi", "Hello", "Greetings"}
        assert set(Structured((1, 2), a=3, b=(4, 5))._flatten()) == {1, 2, 3, 4, 5}
        assert set(Structured((1, Structured(2, b=(3, 4))))._flatten()) == {1, 2, 3, 4}

    def test__simplify(self):
        o = object()
        assert Structured(o)._simplify() is o
        assert Structured(Structured(o))._simplify() is o
        assert Structured(o, key="value")._simplify() == Structured(o, key="value")
        assert Structured(key=Structured(o))._simplify() == Structured(key=o)
        assert Structured(
            key=Structured(o), _metadata={"a": 1}
        )._simplify()._metadata == {"a": 1}

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Cannot simplify `Structured` instances in-place if `unwrap` is `True`."
            ),
        ):
            Structured()._simplify(unwrap=True, inplace=True)

    def test__update(self):
        o = object()
        assert Structured(o)._update([o]) == Structured([o])
        assert Structured()._update(key=o) == Structured(key=o)
        assert Structured(1, key=2)._update(3) == Structured(3, key=2)
        assert Structured(_metadata={"a": 1})._update()._metadata == {"a": 1}

    def test__merge(self):
        _m = Structured._merge

        assert _m() == Structured()
        assert _m(Structured(), Structured()) == Structured()
        assert _m(Structured(["a"]), ["b"]) == Structured(["a", "b"])
        assert _m(Structured(a=1), Structured(b=2)) == Structured(a=1, b=2)
        assert _m(Structured((1, 2, 3)), Structured((4, 5, 6))) == Structured(
            (1, 2, 3, 4, 5, 6)
        )
        assert _m((1, 2, 3), (4, 5, 6)) == Structured((1, 2, 3, 4, 5, 6))

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Substructures for `.` are not aligned and cannot be merged."
            ),
        ):
            _m((1, 2, 3), [4, 5, 6])

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Substructures for `.a.b` are not aligned and cannot be merged."
            ),
        ):
            _m(
                Structured(a=Structured(b=(1, 2, 3))),
                Structured(a=Structured(b=[4, 5, 6])),
            )

        # Test default resolver
        assert _m(Structured(a=[1]), Structured(a=[1])) == Structured(a=[1, 1])
        assert _m(Structured(a={1}), Structured(a={1})) == Structured(a={1})
        assert _m(
            Structured(a={"a": 1, "b": 2}), Structured(a={"b": 3, "c": 4})
        ) == Structured(a={"a": 1, "b": 3, "c": 4})

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "The fallback `merger` for `Structured._merge` does not know how to merge objects of types (<class 'str'>, <class 'str'>). Please specify `merger` explicitly."
            ),
        ):
            _m("asd", "asd")

        # Test custom resolver
        assert True

    def test_mutation(self):
        s = Structured(1)
        s.root = 10
        assert s == Structured(10)
        s.key = 10
        assert s == Structured(10, key=10)
        s["key"] = 20
        assert s == Structured(10, key=20)

        with pytest.raises(AttributeError):
            s._key = 10

        with pytest.raises(KeyError):
            s["_key"] = 10

        with pytest.raises(KeyError):
            s["0invalid"] = 10

        with pytest.raises(KeyError):
            s[0] = 10

        s2 = Structured(Structured((Structured("Hello"),)))
        s2[("root", "root", 0, "b")] = "World"
        assert s2 == Structured(Structured((Structured("Hello", b="World"),)))

        with pytest.raises(KeyError, match="Cannot replace self"):
            s2[()] = "Hello"
        with pytest.raises(
            KeyError, match=re.escape("Object @ ('root', 'root') is not a `Structured`")
        ):
            s2[("root", "root", 2)] = "Hello"

    def test_iteration(self):
        assert list(Structured()) == []
        assert list(Structured("a")) == ["a"]
        assert list(Structured(["a", "b", "c"])) == ["a", "b", "c"]
        assert len(Structured(["a", "b", "c"])) == 3
        assert list(Structured(b="b", c="c")) == ["b", "c"]
        assert list(Structured(c="c", b="b")) == ["c", "b"]
        assert list(Structured("a", b="b", c="c")) == ["a", "b", "c"]
        assert list(Structured("a", b="b", c=["c"])) == ["a", "b", ["c"]]
        assert list(Structured("a", b="b", c=("c",))) == ["a", "b", ("c",)]
        assert len(Structured("a", b="b", c=("c",))) == 3

    def test_equality(self):
        assert Structured() == Structured()
        assert Structured(1) != Structured()
        assert Structured(1) == Structured(1)
        assert Structured(key="value") == Structured(key="value")
        assert Structured(key="value") != Structured(key2="value")
        assert Structured() != object()

    def test_contains(self):
        assert "key" not in Structured()
        assert "key" in Structured(key="hi")

    def test_repr(self):
        assert repr(Structured("a")) == "root:\n    'a'"
        assert repr(Structured("a", b="b")) == "root:\n    'a'\n.b:\n    'b'"
        assert (
            repr(Structured(("a",), b=("b", "c")))
            == "root:\n    [0]:\n        'a'\n.b:\n    [0]:\n        'b'\n    [1]:\n        'c'"
        )

        # Verify that string representations correctly avoid use of `repr`
        assert str(Structured("a")) == "root:\n    a"
        assert str(Structured("a", b="b")) == "root:\n    a\n.b:\n    b"

    def test_pickleable(self):
        o = BytesIO()
        s = Structured("a", b="b", c=("c", "d"))
        pickle.dump(s, o)
        o.seek(0)
        s2 = pickle.load(o)
        assert s2._to_dict() == {
            "root": "a",
            "b": "b",
            "c": ("c", "d"),
        }
