import pytest

from formulaic.parser.types import Factor, Term


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
        assert str(term1 * term2) == "b:c:d"

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
        assert repr(term1) == "b:c"
