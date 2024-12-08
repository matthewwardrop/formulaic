from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from .ordered_set import OrderedSet

if TYPE_CHECKING:
    from .factor import Factor  # pragma: no cover


class Term:
    """
    Represents a "term" of a formula.

    A "term" is a product of "factors" (represented by `Factor`) instances, and
    a formula is made up of a sum of terms.

    Attributes:
        factors: The set of factors to be multiplied to form the term.
        origin: If this `Term` has been derived from another `Term`, for example
            in subformulae, a reference to the original term.
    """

    FACTOR_MATCHER = re.compile(r"(?:^|(?<=:))(`?)(?P<factor>[^`]+?)\1(?=:|$)")

    def __init__(self, factors: Iterable["Factor"], origin: Optional[Term] = None):
        self.factors = tuple(dict.fromkeys(factors))
        self.origin = origin
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
            return self._factor_key == tuple(
                sorted([m.group("factor") for m in self.FACTOR_MATCHER.finditer(other)])
            )
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Term):
            if self.degree == other.degree:
                return sorted(self.factors) < sorted(other.factors)
            if self.degree < other.degree:
                return True
            return False
        return NotImplemented

    def to_terms(
        self, *, context: Optional[Mapping[str, Any]] = None
    ) -> OrderedSet[Term]:
        """
        Convert this `Term` instance into set of `Term`s.
        """
        return OrderedSet((self,))

    def __repr__(self) -> str:
        return ":".join(repr(factor) for factor in self.factors)
