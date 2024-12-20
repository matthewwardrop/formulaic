import pickle
import re
from io import BytesIO

import pandas
import pytest

from formulaic import Formula, SimpleFormula, StructuredFormula
from formulaic.errors import FormulaInvalidError, FormulaMaterializerInvalidError
from formulaic.parser.types.factor import Factor
from formulaic.parser.types.term import Term
from formulaic.utils.structured import Structured


class TestFormula:
    """
    We only test the high-level APIs here; correctness of the formula parsing
    and model matrix materialization is thoroughly tested in other unit tests.
    We also focus mainly on direct `Formula` interactions, rather than its
    subclasses, since those are only used to service the top-level API anyway.
    """

    @pytest.fixture
    def formula_expr(self):
        return Formula("a * b * c")

    @pytest.fixture
    def formula_exprs(self):
        return Formula("a ~ b")

    @pytest.fixture
    def data(self):
        return pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_constructor(self):
        # Empty
        assert isinstance(Formula(), SimpleFormula)
        assert Formula() == []

        # str
        assert Formula("a + b") == ["1", "a", "b"]

        # List[str | Term]
        assert Formula(["a", "c", Term([Factor("b")]), "1"]) == [
            "1",
            "a",
            "c",
            "b",
        ]

        # Set[str | Term]
        assert sorted(Formula({"a", Term([Factor("b")]), "c"})) == ["a", "b", "c"]

        # Dict[str, FormulaSpec]
        assert Formula(
            {"root": "a", "nested": "a", "nested2": ["a", "b"]}
        )._to_dict() == {
            "root": ["1", "a"],
            "nested": ["a"],
            "nested2": ["a", "b"],
        }

        # Tuple[List[str], ...]
        f = Formula((["a", "b"], ["c", "d"]))
        assert isinstance(f, StructuredFormula)
        assert isinstance(f.root, tuple)
        assert [str(t) for t in f[0]] == ["a", "b"]
        assert [str(t) for t in f[1]] == ["c", "d"]

        # Tuple[str, List[str]]
        f = Formula(("a", ["b", "c"]))
        assert isinstance(f, StructuredFormula)
        assert f._has_structure
        assert f[0] == ["1", "a"]
        assert f[1] == ["b", "c"]

        # Formula
        f = Formula(["a"])
        assert isinstance(f, SimpleFormula)
        assert isinstance(Formula(f), SimpleFormula)
        assert Formula(f) == f
        assert Formula.from_spec(f) is f
        assert Formula.from_spec(["a"]) == f

        # Test wildcards
        assert Formula(
            ".", _context={"__formulaic_variables_available__": ["a", "b"]}
        ) == ["1", "a", "b"]
        assert Formula(
            "a ~ .", _context={"__formulaic_variables_available__": ["a", "b"]}
        )._to_dict() == {"lhs": ["a"], "rhs": ["1", "b"]}

    def test_terms(self, formula_expr):
        assert [str(t) for t in formula_expr] == [
            "1",
            "a",
            "b",
            "c",
            "a:b",
            "a:c",
            "b:c",
            "a:b:c",
        ]

    def test_ordering(self):
        assert [str(t) for t in Formula("a+e:d+b:c+f")] == [
            "1",
            "a",
            "f",
            "e:d",
            "b:c",
        ]
        assert [str(t) for t in Formula("a+e:d+b:c+f", _ordering="degree")] == [
            "1",
            "a",
            "f",
            "e:d",
            "b:c",
        ]
        assert [str(t) for t in Formula("a+e:d+b:c+f", _ordering="none")] == [
            "1",
            "a",
            "e:d",
            "b:c",
            "f",
        ]
        assert [str(t) for t in Formula("a+e:d+b:c+f", _ordering="sort")] == [
            "1",
            "a",
            "f",
            "b:c",
            "d:e",
        ]

        # Test nested ordering

        assert [str(t) for t in Formula("y~a+e:d+b:c+f").rhs] == [
            "1",
            "a",
            "f",
            "e:d",
            "b:c",
        ]
        assert [str(t) for t in Formula("y~a+e:d+b:c+f", _ordering="degree").rhs] == [
            "1",
            "a",
            "f",
            "e:d",
            "b:c",
        ]
        assert [str(t) for t in Formula("y~a+e:d+b:c+f", _ordering="none").rhs] == [
            "1",
            "a",
            "e:d",
            "b:c",
            "f",
        ]
        assert [str(t) for t in Formula("y~a+e:d+b:c+f", _ordering="sort").rhs] == [
            "1",
            "a",
            "f",
            "b:c",
            "d:e",
        ]

    def test_get_model_matrix(self, formula_expr, formula_exprs, data):
        mm_expr = formula_expr.get_model_matrix(data)
        mm_exprs = formula_exprs.get_model_matrix(data, materializer="pandas")

        assert mm_expr.shape == (3, 8)
        assert isinstance(mm_exprs, Structured) and len(mm_exprs) == 2

    def test_structured(self, formula_exprs):
        assert formula_exprs.lhs == ["a"]
        assert formula_exprs.rhs == ["1", "b"]
        assert Formula("a | b")[0] == ["1", "a"]
        assert isinstance(Formula(["a"], b=["b"])["root"], Formula)
        assert isinstance(formula_exprs["lhs"], Formula)

        with pytest.raises(
            AttributeError,
            match=re.escape(
                "This `StructuredFormula` instance does not have structure @ `'missing'`."
            ),
        ):
            formula_exprs.missing
        with pytest.raises(
            KeyError,
            match=re.escape(
                "This `StructuredFormula` instance does not have structure @ `0`."
            ),
        ):
            formula_exprs[0]

    def test_differentiate(self):
        f = Formula("a + b + log(c) - 1")
        assert f.differentiate("a") == ["1", "0", "0"]
        assert f.differentiate("c") == ["0", "0", "0"]

        g = Formula("a:b + b:c + c:d - 1")
        assert g.differentiate("b") == ["a", "c", "0"]  # order preserved

    def test_differentiate_with_sympy(self):
        pytest.importorskip("sympy")
        f = Formula("a + b + log(c) - 1")
        assert f.differentiate("c", use_sympy=True) == ["0", "0", "(1/c)"]

        g = Formula("y ~ log(x)")
        assert g.differentiate("x", use_sympy=True)._to_dict() == {
            "lhs": ["0"],
            "rhs": ["0", "(1/x)"],
        }

        h = Formula("a + {a**2} + b - 1").differentiate("a", use_sympy=True)
        assert h == ["1", "(2*a)", "0"]  # order preserved

    def test_repr(self, formula_expr, formula_exprs):
        assert repr(formula_expr) == "1 + a + b + c + a:b + a:c + b:c + a:b:c"
        assert repr(formula_exprs) == ".lhs:\n    a\n.rhs:\n    1 + b"
        assert repr(Formula("a | b")) == (
            "root:\n    [0]:\n        1 + a\n    [1]:\n        1 + b"
        )
        assert str(formula_expr) == "1 + a + b + c + a:b + a:c + b:c + a:b:c"
        assert str(formula_exprs) == ".lhs:\n    a\n.rhs:\n    1 + b"

    def test_equality(self):
        assert Formula("a + b") == Formula("a+b")
        assert Formula("a + b") != 1

    def test_invalid_formula(self):
        with pytest.raises(FormulaInvalidError):
            Formula(None)
        with pytest.raises(FormulaInvalidError):
            Formula({"a": 1, "b": 2})
        with pytest.raises(FormulaInvalidError):
            Formula([{"a": 1}])
        with pytest.raises(FormulaInvalidError):
            # Should not be possible to reach this, but check anyway.
            SimpleFormula._SimpleFormula__validate_terms(("a",))

    def test_invalid_materializer(self, formula_expr, data):
        with pytest.raises(FormulaMaterializerInvalidError):
            formula_expr.get_model_matrix(data, materializer=object())

    def test_pickleable(self, formula_exprs):
        o = BytesIO()
        pickle.dump(formula_exprs, o)
        o.seek(0)
        formula = pickle.load(o)
        assert formula.lhs == ["a"]
        assert formula.rhs == ["1", "b"]

    def test_required_variables(self):
        assert Formula("a + b").required_variables == {"a", "b"}
        assert Formula("a + b + a:b").required_variables == {"a", "b"}
        assert Formula("a + C(b)").required_variables == {"a", "b"}
        assert Formula("a + C(b, levels=['b1', 'b2'])").required_variables == {
            "a",
            "b",
        }
        assert Formula("a + C(b, contr.Treatment)").required_variables == {
            "a",
            "b",
        }
        assert Formula("y ~ x").required_variables == {"x", "y"}


class TestSimpleFormula:
    """
    We do not expect people to directly interact with `SimpleFormula`
    constructors very often, but let's make sure all the expected safe-guards
    are in place.
    """

    def test_constructor(self):
        assert SimpleFormula([Term((Factor("a"), Factor("b")))]) == ["a:b"]

        with pytest.raises(FormulaInvalidError):
            SimpleFormula(None)
        with pytest.raises(
            FormulaInvalidError,
            match=re.escape(
                "`SimpleFormula` should be constructed from a list of `Term` instances"
            ),
        ):
            SimpleFormula("a")
        with pytest.raises(
            FormulaInvalidError,
            match=re.escape(
                "All components of a `SimpleFormula` should be `Term` instances."
            ),
        ):
            SimpleFormula(["a", "b"])
        with pytest.raises(
            FormulaInvalidError,
            match=re.escape("`SimpleFormula` does not support nested structure."),
        ):
            SimpleFormula(nested=["a"])

    def test_sequencing(self):
        f = SimpleFormula([Term((Factor("a"), Factor("b"))), Term((Factor("c"),))])
        assert f == ["c", "a:b"]
        assert f[0] == "c"
        assert f[1] == "a:b"
        assert f[0:1] == ["c"]

    def test_mutation(self):
        f = SimpleFormula([Term((Factor("a"), Factor("b"))), Term((Factor("c"),))])
        f[0] = Term((Factor("d"),))
        assert f == ["d", "a:b"]
        f[0] = Term([Factor(f) for f in "abcd"])
        assert f == ["a:b", "a:b:c:d"]

        g = SimpleFormula(
            [Term((Factor("a"), Factor("b"))), Term((Factor("c"),))], _ordering="none"
        )
        g[0] = Term((Factor("d"),))
        assert g == ["d", "c"]
        g[0] = Term([Factor(f) for f in "abcd"])
        assert g == ["a:b:c:d", "c"]

        with pytest.raises(
            FormulaInvalidError,
            match=re.escape(
                "All components of a `SimpleFormula` should be `Term` instances."
            ),
        ):
            f[0] = "a"

    def test_insertion_and_deletion(self):
        f = SimpleFormula([Term((Factor("a"), Factor("b"))), Term((Factor("c"),))])
        f.insert(0, Term([Factor(f) for f in "abcd"]))
        assert f == [
            "c",
            "a:b",
            "a:b:c:d",
        ]
        del f[-1]
        assert f == [
            "c",
            "a:b",
        ]

        g = SimpleFormula(
            [Term((Factor("a"), Factor("b"))), Term((Factor("c"),))], _ordering="none"
        )
        g.insert(0, Term([Factor(f) for f in "abcd"]))
        assert g == [
            "a:b:c:d",
            "a:b",
            "c",
        ]
        del g[-1]
        assert g == [
            "a:b:c:d",
            "a:b",
        ]

        with pytest.raises(
            FormulaInvalidError,
            match=re.escape(
                "All components of a `SimpleFormula` should be `Term` instances."
            ),
        ):
            f.insert(0, "a")

    def test_deprecated_methods(self):
        f = Formula("a + b")

        with pytest.warns(DeprecationWarning):
            assert f.root is f

        with pytest.warns(DeprecationWarning):
            assert f._has_root is True

        with pytest.warns(DeprecationWarning):
            assert f._has_structure is False

        with pytest.warns(DeprecationWarning):
            assert f._map(lambda x: x) is f

        with pytest.warns(DeprecationWarning):
            assert f._map(lambda x, ctx: x) is f

        with pytest.warns(DeprecationWarning):
            assert next(iter(f._flatten())) is f

        with pytest.warns(DeprecationWarning):
            assert f._to_dict() == {"root": f}

        with pytest.warns(DeprecationWarning):
            assert f._simplify() is f

        with pytest.warns(DeprecationWarning):
            assert f._update(nested="a") == StructuredFormula(f, nested="a")


class TestStructuredFormula:
    def test_pickling(self):
        s = StructuredFormula("a + b", _context={})
        s2 = pickle.loads(pickle.dumps(s))
        assert s == s2
        assert s2._context is None
