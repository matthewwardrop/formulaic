import pytest

import pandas

from formulaic import model_matrix
from formulaic.errors import FactorEvaluationError


def global_test(x):
    return x**2


class TestSugar:
    @pytest.fixture
    def data(self):
        return pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_model_matrix(self, data):
        def local_test(x):
            return x**2

        r = model_matrix("0 + global_test(a) + local_test(b)", data)
        assert list(r["global_test(a)"]) == [1, 4, 9]
        assert list(r["local_test(b)"]) == [16, 25, 36]

        # reuse model spec from previously generated model matrix.
        r2 = model_matrix(r, data)
        assert list(r2["global_test(a)"]) == [1, 4, 9]
        assert list(r2["local_test(b)"]) == [16, 25, 36]

        with pytest.raises(FactorEvaluationError):
            model_matrix("0 + global_test(a) + local_test(b)", data, context=None)
