import pickle
import re
from io import BytesIO

import pytest

from formulaic.parser.types import Structured


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

    def test_mapped_attrs(self):
        class O:
            def __init__(self, o):
                self.o = o

        assert (
            Structured(O(1), a=O(2), b=(O(3), O(4)), _mapped_attrs={"o"}).o._to_dict()
            == Structured(1, a=2, b=(3, 4))._to_dict()
        )

    def test__map(self):
        assert Structured("Hi", a="Hello", b="Greetings")._map(len)._to_dict() == {
            None: 2,
            "a": 5,
            "b": 9,
        }
        assert Structured(("Hi", "Dave"), a=["Response", "not", "forthcoming"])._map(
            len
        )._to_dict() == {
            None: (2, 4),
            "a": 3,
        }
        assert Structured(("Hi", Structured("Hi!")), a=Structured(hello="world"))._map(
            len
        )._to_dict() == {
            None: (2, 1),
            "a": {
                "hello": 5,
            },
        }

    def test_iteration(self):
        assert list(Structured()) == []
        assert list(Structured("a")) == ["a"]
        assert list(Structured(b="b", c="c")) == ["b", "c"]
        assert list(Structured(c="c", b="b")) == ["c", "b"]
        assert list(Structured("a", b="b", c="c")) == ["a", "b", "c"]
        assert list(Structured("a", b="b", c=["c"])) == ["a", "b", ["c"]]
        assert list(Structured("a", b="b", c=("c",))) == ["a", "b", ("c",)]

    def test_repr(self):
        assert repr(Structured("a")) == ("root:\n" "    'a'")
        assert repr(Structured("a", b="b")) == (
            "root:\n" "    'a'\n\n" "Structured children:\n\n" ".b\n" "    'b'"
        )
        assert repr(Structured(("a",), b=("b", "c"))) == (
            "root:\n"
            "    [0]:\n"
            "        'a'\n\n"
            "Structured children:\n\n"
            ".b\n"
            "    [0]:\n"
            "        'b'\n"
            "    [1]:\n"
            "        'c'"
        )

        # Verify that string representations correctly avoid use of `repr`
        assert str(Structured("a")) == ("root:\n" "    a")
        assert str(Structured("a", b="b")) == (
            "root:\n" "    a\n\n" "Structured children:\n\n" ".b\n" "    b"
        )

    def test_pickleable(self):
        o = BytesIO()
        s = Structured("a", b="b", c=("c", "d"))
        pickle.dump(s, o)
        o.seek(0)
        s2 = pickle.load(o)
        assert s2._to_dict() == {
            None: "a",
            "b": "b",
            "c": ("c", "d"),
        }
