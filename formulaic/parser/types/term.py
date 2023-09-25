from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .factor import Factor  # pragma: no cover


class Term:
    """
    Represents a "term" of a formula.

    A "term" is a product of "factors" (represented by `Factor`) instances, and
    a formula is made up of a sum of terms.

    Attributes:
        factors: The set of factors to be multiplied to form the term.
    """

    def __init__(self, factors: Iterable["Factor"]):
        self.factors = tuple(dict.fromkeys(factors))
        self._factor_key = tuple(factor.expr for factor in sorted(self.factors))
        self._hash = hash(":".join(self._factor_key))

    @property
    def degree(self) -> int:
        """
        The degree of the `Term`. Literal factors do not contribute to the degree.
        """
        return len(
            tuple(f for f in self.factors if f.eval_method != f.eval_method.LITERAL)
        )

    # Transforms and comparisons

    def __mul__(self, other: Any) -> Term:
        if isinstance(other, Term):
            return Term([*self.factors, *other.factors])
        return NotImplemented

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Term):
            return self._factor_key == other._factor_key
        if isinstance(other, str):
            return self._factor_key == tuple(sorted(other.split(":")))
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Term):
            if self.degree == other.degree:
                return sorted(self.factors) < sorted(other.factors)
            if self.degree < other.degree:
                return True
            return False
        return NotImplemented

    def __repr__(self) -> str:
        return ":".join(factor.expr for factor in self.factors)
