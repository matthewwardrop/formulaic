import pickle
import re
from io import BytesIO

import pandas
import pytest

from formulaic import Formula
from formulaic.errors import FormulaInvalidError, FormulaMaterializerInvalidError
from formulaic.parser.types import Structured


class TestFormula:
    """
    We only test the high-level APIs here; correctness of the formula parsing
    and model matrix materialization is thoroughly tested in other unit tests.
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
        assert [str(t) for t in Formula(["a", "b", "c"])] == ["a", "b", "c"]
        assert [str(t) for t in Formula(["a", "c", "b", "1"])] == [
            "1",
            "a",
            "c",
            "b",
        ]

        f = Formula((["a", "b"], ["c", "d"]))
        assert isinstance(f, Structured)
        assert isinstance(f.root, tuple)
        assert [str(t) for t in f[0]] == ["a", "b"]
        assert [str(t) for t in f[1]] == ["c", "d"]

        f = Formula(("a", ["b", "c"]))
        assert f._has_structure
        assert f[0].root == ["1", "a"]
        assert f[1].root == ["b", "c"]

        f = Formula(["a"])
        assert Formula.from_spec(f) is f
        assert Formula.from_spec(["a"]) == f

        f2 = Formula(f)
        assert f2 == f

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
        # RHS variable order intentionally non-lexicographic to test ordering
        assert [str(t) for t in Formula("f+a+e:d+b:c", _ordering="grouped")] == [
            "1",
            "f",
            "a",
            "e:d",
            "b:c",
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
        # RHS variable order intentionally non-lexicographic to test ordering
        assert [str(t) for t in Formula("y~f+a+e:d+b:c", _ordering="grouped").rhs] == [
            "1",
            "f",
            "a",
            "e:d",
            "b:c",
        ]

    def test_get_model_matrix(self, formula_expr, formula_exprs, data):
        mm_expr = formula_expr.get_model_matrix(data)
        mm_exprs = formula_exprs.get_model_matrix(data, materializer="pandas")

        assert mm_expr.shape == (3, 8)
        assert isinstance(mm_exprs, Structured) and len(mm_exprs) == 2

    def test_structured(self, formula_exprs):
        assert formula_exprs.lhs.root == ["a"]
        assert formula_exprs.rhs.root == ["1", "b"]
        assert Formula("a | b")[0].root == ["1", "a"]
        assert isinstance(Formula(["a"], b=["b"])["root"], list)
        assert isinstance(formula_exprs["lhs"], Formula)

        with pytest.raises(
            AttributeError,
            match=re.escape(
                "This `Formula` instance does not have structure @ `'missing'`."
            ),
        ):
            formula_exprs.missing
        with pytest.raises(
            KeyError,
            match=re.escape("This `Formula` instance does not have structure @ `0`."),
        ):
            formula_exprs[0]

    def test_differentiate(self):
        f = Formula("a + b + log(c) - 1")
        assert f.differentiate("a").root == ["1", "0", "0"]
        assert f.differentiate("c").root == ["0", "0", "0"]

    def test_differentiate_with_sympy(self):
        pytest.importorskip("sympy")
        f = Formula("a + b + log(c) - 1")
        assert f.differentiate("c", use_sympy=True).root == ["0", "0", "(1/c)"]

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
            Formula._Formula__validate_terms(("a",))

    def test_invalid_materializer(self, formula_expr, data):
        with pytest.raises(FormulaMaterializerInvalidError):
            formula_expr.get_model_matrix(data, materializer=object())

    def test_pickleable(self, formula_exprs):
        o = BytesIO()
        pickle.dump(formula_exprs, o)
        o.seek(0)
        formula = pickle.load(o)
        assert formula.lhs.root == ["a"]
        assert formula.rhs.root == ["1", "b"]

    def test_required_variables(self):
        assert Formula("a + b").required_variables == {"a", "b"}
        assert Formula("a + b + a:b").required_variables == {"a", "b"}
        assert Formula("a + C(b)").required_variables == {"a", "b"}
        assert Formula("a + C(b, levels=['b1', 'b2'])").required_variables == {"a", "b"}
        assert Formula("a + C(b, contr.Treatment)").required_variables == {"a", "b"}
