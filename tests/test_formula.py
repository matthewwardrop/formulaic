import pytest

import pandas

from formulaic import Formula
from formulaic.errors import FormulaInvalidError, FormulaMaterializerInvalidError


class TestFormula:
    """
    We only test the high-level APIs here; correctness of the formula parsing
    and model matrix materialization is thoroughly tested in other unit tests.
    """

    @pytest.fixture
    def formula_expr(self):
        return Formula('a * b * c')

    @pytest.fixture
    def formula_exprs(self):
        return Formula('a ~ b ~ c')

    @pytest.fixture
    def data(self):
        return pandas.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

    def test_constructor(self):
        assert [str(t) for t in Formula(['a', 'b', 'c']).terms] == ['a', 'b', 'c']
        assert [str(t) for t in Formula(['a', 'c', 'b', '1']).terms] == ['a', 'c', 'b', '1']

        f = Formula((['a', 'b'], ['c', 'd']))
        assert isinstance(f.terms, tuple)
        assert [str(t) for t in f.terms[0]] == ['a', 'b']
        assert [str(t) for t in f.terms[1]] == ['c', 'd']

        f = Formula(['a'])
        assert Formula.from_spec(f) is f
        assert Formula.from_spec(['a']) == f

    def test_terms(self, formula_expr, formula_exprs):
        assert [str(t) for t in formula_expr.terms] == ['1', 'a', 'b', 'c', 'a:b', 'a:c', 'b:c', 'a:b:c']
        assert tuple([str(t) for t in tg] for tg in formula_exprs.terms) == (['a'], ['b'], ['1', 'c'])

    def test_get_model_matrix(self, formula_expr, formula_exprs, data):
        mm_expr = formula_expr.get_model_matrix(data)
        mm_exprs = formula_exprs.get_model_matrix(data, materializer='pandas')

        assert mm_expr.shape == (3, 8)
        assert isinstance(mm_exprs, tuple) and len(mm_exprs) == 3

    def test_differentiate(self):
        f = Formula('a + b + log(c) - 1')
        assert f.differentiate('a').terms == ['1', '0', '0']
        assert f.differentiate('c').terms == ['0', '0', '0']
        assert f.differentiate('c', use_sympy=True).terms == ['0', '0', '(1/c)']

    def test_repr(self, formula_expr, formula_exprs):
        assert repr(formula_expr) == '1 + a + b + c + a:b + a:c + b:c + a:b:c'
        assert repr(formula_exprs) == 'a ~ b ~ 1 + c'

    def test_equality(self):
        assert Formula('a + b') == Formula('a+b')
        assert Formula('a + b') != 1

    def test_invalid_formula(self):
        with pytest.raises(FormulaInvalidError):
            Formula(None)
        with pytest.raises(FormulaInvalidError):
            Formula({'a': 1, 'b': 2})
        with pytest.raises(FormulaInvalidError):
            Formula([{'a': 1}])

    def test_invalid_materializer(self, formula_expr, data):
        with pytest.raises(FormulaMaterializerInvalidError):
            formula_expr.get_model_matrix(data, materializer=object())


    def test_bs(self):
        df = pandas.DataFrame({
            'y': [0, 1, 2],
            'a': ['A', 'B', 'C'],
            'b': [0.3, 0.1, 0.2],
        })
        y, X = Formula("y ~ bs(b, n_knots=5)").get_model_matrix(df)
        assert len(X.columns) == 6

