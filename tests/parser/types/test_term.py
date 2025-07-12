import pytest

from formulaic.parser.types import Factor, Term
from formulaic.utils.ordered_set import OrderedSet


class TestTerm:
    @pytest.fixture
    def term1(self):
        return Term([Factor("c"), Factor("b")])

    @pytest.fixture
    def term2(self):
        return Term([Factor("c"), Factor("d")])

    @pytest.fixture
    def term3(self):
        return Term([Factor("a"), Factor("b"), Factor("c")])

    def test_mul(self, term1, term2):
        assert str(term1 * term2) == "c:b:d"

        with pytest.raises(TypeError):
            term1 * 1

    def test_hash(self, term1):
        assert hash(term1) == hash("b:c")

    def test_equality(self, term1, term2):
        assert term1 == term1
        assert term1 == "b:c"
        assert term1 != term2
        assert term1 != 1

    def test_sort(self, term1, term2, term3):
        assert term1 < term2
        assert term2 < term3
        assert term1 < term3
        assert not (term3 < term1)

        with pytest.raises(TypeError):
            term1 < 1

    def test_repr(self, term1):
        assert repr(term1) == "c:b"

    def test_degree(self, term1, term3):
        assert term1.degree == 2
        assert term3.degree == 3
        assert Term([Factor("1", eval_method="literal")]).degree == 0
        assert Term([Factor("1", eval_method="literal"), Factor("x")]).degree == 1

    def test_to_terms(self, term1):
        assert term1.to_terms() == OrderedSet((term1,))
