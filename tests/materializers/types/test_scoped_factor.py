import pytest

from formulaic.materializers.types import ScopedFactor
from formulaic.parser.types import Factor


class TestScopedFactor:
    @pytest.fixture
    def scoped_factor(self):
        return ScopedFactor(Factor("a"))

    @pytest.fixture
    def scoped_factor_reduced(self):
        return ScopedFactor(Factor("a"), reduced=True)

    def test_repr(self, scoped_factor, scoped_factor_reduced):
        assert repr(scoped_factor) == "a"
        assert repr(scoped_factor_reduced) == "a-"

    def test_hash(self, scoped_factor, scoped_factor_reduced):
        assert hash(scoped_factor) == hash("a")
        assert hash(scoped_factor_reduced) == hash("a-")

    def test_equality(self, scoped_factor, scoped_factor_reduced):
        assert scoped_factor == scoped_factor
        assert scoped_factor != scoped_factor_reduced
        assert scoped_factor != 1

    def test_sort(self, scoped_factor, scoped_factor_reduced):
        assert scoped_factor_reduced < scoped_factor
        assert scoped_factor < ScopedFactor(Factor("b"))

        with pytest.raises(TypeError):
            scoped_factor < 1
