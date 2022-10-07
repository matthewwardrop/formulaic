from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .factor import Factor  # pragma: no cover


class Term:
    """
    Represents a "term" of a formula.

    A "term" is a product of "factors" (represented by `Factor`) instances, and
    a formula is made up of a sum of terms.

    Attributes:
        factors: The set of factors to be multipled to form the term.
        preserve_rank: Whether to preserve the term structure even when
            `ensure_full_rank` is specified. Other terms without this set may
            still be affected by the presence of this term.
    """

    def __init__(self, factors: Iterable["Factor"], preserve_rank: bool = False):
        self.factors = tuple(sorted(set(factors)))
        self.preserve_rank = preserve_rank
        self._factor_exprs = tuple(factor.expr for factor in self.factors)
        self._hash = hash(repr(self))

    # Transforms and comparisons

    def __mul__(self, other):
        if isinstance(other, Term):
            return Term([*self.factors, *other.factors])
        return NotImplemented

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Term):
            return self._factor_exprs == other._factor_exprs
        if isinstance(other, str):
            return repr(self) == other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Term):
            if len(self.factors) == len(other.factors):
                return sorted(self.factors) < sorted(other.factors)
            if len(self.factors) < len(other.factors):
                return True
            return False
        return NotImplemented

    def __repr__(self):
        return ":".join(self._factor_exprs)


class TermGroup(Term):
    """
    Represents a group randomized term a formula.

    Attributes:
        term:
        group:
        joiner:
    """

    def __init__(self, term, group, joiner="|"):
        self.term = term
        self.group = group
        self.joiner = joiner
        super().__init__(factors=[*term.factors, *group.factors], preserve_rank=True)

    def __repr__(self):
        return repr(self.term) + self.joiner + repr(self.group)
