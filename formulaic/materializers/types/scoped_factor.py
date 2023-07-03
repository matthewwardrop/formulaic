from __future__ import annotations

from typing import Any

from .evaluated_factor import EvaluatedFactor


class ScopedFactor:
    """
    A wrapper around an `EvaluatedFactor` instance that indicates whether or not the
    Factor should be materialized in full- or reduced-rank.
    """

    def __init__(self, factor: EvaluatedFactor, reduced: bool = False) -> None:
        self.factor = factor
        self.reduced = reduced

    def __repr__(self) -> str:
        return repr(self.factor) + ("-" if self.reduced else "")

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ScopedFactor):
            return self.factor == other.factor and self.reduced == other.reduced
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, ScopedFactor):
            if self.factor == other.factor:
                return self.reduced > other.reduced
            return self.factor < other.factor
        return NotImplemented
