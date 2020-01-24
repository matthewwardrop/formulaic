import numpy
import pandas
import pytest
import scipy.sparse as spsparse

from formulaic.errors import FactorEncodingError, FactorEvaluationError
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.types import EvaluatedFactor
from formulaic.parser.types import Factor


PANDAS_TESTS = {
    'a': (['Intercept', 'a'], ['Intercept', 'a']),
    'A': (['Intercept', 'A[T.b]', 'A[T.c]'], ['Intercept', 'A[T.a]', 'A[T.b]', 'A[T.c]']),
    'C(A)': (['Intercept', 'C(A)[T.b]', 'C(A)[T.c]'], ['Intercept', 'C(A)[T.a]', 'C(A)[T.b]', 'C(A)[T.c]']),
    'a:A': (['Intercept', 'A[T.a]:a', 'A[T.b]:a', 'A[T.c]:a'], ['Intercept', 'A[T.a]:a', 'A[T.b]:a', 'A[T.c]:a']),
}


class TestPandasMaterializer:

    @pytest.fixture
    def data(self):
        return pandas.DataFrame({'a': [1, 2, 3], 'A': ['a', 'b', 'c']})

    @pytest.fixture
    def materializer(self, data):
        return PandasMaterializer(data)

    @pytest.fixture
    def materializer_sparse(self, data):
        return PandasMaterializer(data, sparse=True)

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
    def test_get_model_matrix(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.columns) == tests[0]

        mm = materializer.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, pandas.DataFrame)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.columns) == tests[1]

    @pytest.mark.parametrize("formula,tests", PANDAS_TESTS.items())
    def test_get_model_matrix_sparse(self, materializer_sparse, formula, tests):
        mm = materializer_sparse.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.model_spec.feature_names) == tests[0]

        mm = materializer_sparse.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, spsparse.csc_matrix)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.model_spec.feature_names) == tests[1]

    def test_state(self, materializer):
        mm = materializer.get_model_matrix('center(a) - 1')
        assert isinstance(mm, pandas.DataFrame)
        assert list(mm.columns) == ['center(a)']
        assert numpy.allclose(mm['center(a)'], [-1, 0, 1])

        mm2 = PandasMaterializer(pandas.DataFrame({'a': [4, 5, 6]})).get_model_matrix(mm.model_spec)
        assert isinstance(mm2, pandas.DataFrame)
        assert list(mm2.columns) == ['center(a)']
        assert numpy.allclose(mm2['center(a)'], [2, 3, 4])

        mm3 = mm.model_spec.get_model_matrix(pandas.DataFrame({'a': [4, 5, 6]}))
        assert isinstance(mm3, pandas.DataFrame)
        assert list(mm3.columns) == ['center(a)']
        assert numpy.allclose(mm3['center(a)'], [2, 3, 4])

    def test_factor_evaluation_edge_cases(self, materializer):
        # Test that categorical kinds are set if type would otherwise be numerical
        ev_factor = materializer._evaluate_factor(Factor('a', eval_method='lookup', kind='categorical'), {}, {})
        assert ev_factor.kind.value == 'categorical'

        # Test that other kind mismatches result in an exception
        materializer.factor_cache = {}
        with pytest.raises(FactorEncodingError):
            materializer._evaluate_factor(Factor('A', eval_method='lookup', kind='numerical'), {}, {})

        # Test that if an encoding has already been determined, that an exception is raised
        # if the new encoding does not match
        materializer.factor_cache = {}
        with pytest.raises(FactorEncodingError):
            materializer._evaluate_factor(Factor('a', eval_method='lookup', kind='numerical'), {}, {'a': ('categorical', {})})

        # Test that invalid (kind == UNKNOWN) factors raise errors
        materializer.factor_cache = {}
        with pytest.raises(FactorEvaluationError):
            assert materializer._evaluate_factor(Factor('a'), {}, {})

    def test_categorical_dict_detection(self, materializer):
        assert materializer._is_categorical({'__kind__': 'categorical'})

    def test_encoding_edge_cases(self, materializer):
        # Verify that constant encoding works well
        assert (
            list(
                materializer._encode_evaled_factor(
                    factor=EvaluatedFactor(
                        Factor("10", eval_method='literal', kind='constant'),
                        values=10,
                        kind='constant',
                    ),
                    encoder_state={},
                )['10']
            ) == [10, 10, 10]
        )
