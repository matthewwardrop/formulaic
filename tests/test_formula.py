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
        assert [str(t) for t in Formula(["a", "b", "c"]).terms] == ["a", "b", "c"]
        assert [str(t) for t in Formula(["a", "c", "b", "1"]).terms] == [
            "a",
            "c",
            "b",
            "1",
        ]

        f = Formula((["a", "b"], ["c", "d"]))
        assert isinstance(f.terms, tuple)
        assert [str(t) for t in f.terms[0]] == ["a", "b"]
        assert [str(t) for t in f.terms[1]] == ["c", "d"]

        f = Formula(["a"])
        assert Formula.from_spec(f) is f
        assert Formula.from_spec(["a"]) == f

    def test_terms(self, formula_expr):
        assert [str(t) for t in formula_expr.terms] == [
            "1",
            "a",
            "b",
            "c",
            "a:b",
            "a:c",
            "b:c",
            "a:b:c",
        ]

    def test_get_model_matrix(self, formula_expr, formula_exprs, data):
        mm_expr = formula_expr.get_model_matrix(data)
        mm_exprs = formula_exprs.get_model_matrix(data, materializer="pandas")

        assert mm_expr.shape == (3, 8)
        assert isinstance(mm_exprs, Structured) and len(mm_exprs) == 2

    def test_structured(self, formula_exprs):
        assert formula_exprs.lhs.terms == ["a"]
        assert formula_exprs.rhs.terms == ["1", "b"]
        assert Formula("a | b")[0].terms == ["1", "a"]

        with pytest.raises(
            AttributeError,
            match=re.escape("This formula has no substructures keyed by 'missing'."),
        ):
            formula_exprs.missing
        with pytest.raises(
            KeyError,
            match=re.escape(
                "This formula does not have any sub-parts indexable via `0`."
            ),
        ):
            formula_exprs[0]

    def test_differentiate(self):
        f = Formula("a + b + log(c) - 1")
        assert f.differentiate("a").terms == ["1", "0", "0"]
        assert f.differentiate("c").terms == ["0", "0", "0"]
        assert f.differentiate("c", use_sympy=True).terms == ["0", "0", "(1/c)"]

    def test_repr(self, formula_expr, formula_exprs):
        assert repr(formula_expr) == "1 + a + b + c + a:b + a:c + b:c + a:b:c"
        assert repr(formula_exprs) == (".lhs\n" "    a\n" ".rhs\n" "    1 + b")
        assert repr(Formula("a | b")) == (
            "root:\n" "    [0]:\n" "        1 + a\n" "    [1]:\n" "        1 + b"
        )

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

    def test_invalid_materializer(self, formula_expr, data):
        with pytest.raises(FormulaMaterializerInvalidError):
            formula_expr.get_model_matrix(data, materializer=object())

    def test_pickleable(self, formula_exprs):
        o = BytesIO()
        pickle.dump(formula_exprs, o)
        o.seek(0)
        formula = pickle.load(o)
        assert formula.lhs.terms == ["a"]
        assert formula.rhs.terms == ["1", "b"]
