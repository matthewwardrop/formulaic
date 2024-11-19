from __future__ import annotations

from typing import Any, Iterable, Mapping, Set

from formulaic.materializers.types.evaluated_factor import EvaluatedFactor
from formulaic.utils.variables import Variable

from .scoped_factor import ScopedFactor


class ScopedTerm:
    """
    A representation of a `Term` where scopes have been applied to the factors.
    Recall that "scopes" in this context refer to whether or not the factor
    values have been reduced in rank or not.

    Attributes:
        factors: The `ScopedFactor` instances.
        scale: The global factor associated with the term that should be applied
            during column materialization.
    """

    __slots__ = ("factors", "scale")

    def __init__(self, factors: Iterable[ScopedFactor], scale: float = 1):
        self.factors = tuple(dict.fromkeys(factors))
        self.scale = scale

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.factors)))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ScopedTerm):
            return sorted(self.factors) == sorted(other.factors)
        return NotImplemented

    def __repr__(self) -> str:
        factor_repr = (
            ":".join(f.__repr__() for f in self.factors) if self.factors else "1"
        )
        if self.scale is not None and self.scale != 1:
            return f"{self.scale}*{factor_repr}"
        return factor_repr

    def copy(self, *, without_values: bool = False) -> ScopedTerm:
        """
        Return a copy of this `ScopedTerm` instance, potentially without the
        materialized values attached. This is used during the generation of
        model specs in order to reduce the size of the serialized specs.

        Args:
            without_values: Whether the materialized values should be omitted
                from the copy.
        """
        factors: Iterable[ScopedFactor] = self.factors
        if without_values:
            factors = [
                ScopedFactor(
                    factor=factor.factor.replace(values=None),
                    reduced=factor.reduced,
                )
                for factor in factors
            ]
        return ScopedTerm(factors, scale=self.scale)

    def rehydrate(self, factor_values: Mapping[str, EvaluatedFactor]) -> ScopedTerm:
        """
        Rehydrate the `ScopedTerm` instance with new `EvaluatedFactor` instances
        that have been computed for each factor. An exception will be raised if
        any required factor is missing from the `factor_values` mapping. This is
        used to allow the reuse of serialized `ScopedTerm` structure that have
        had their factor values excised using `.copy(without_values=True)`.

        Args:
            factor_values: A mapping from factor expressions to `EvaluatedFactor`
                instances. This mapping should contain all factors in the term.
        """
        return ScopedTerm(
            [
                ScopedFactor(
                    factor=factor_values[factor.factor.expr],
                    reduced=factor.reduced,
                )
                for factor in self.factors
            ],
            scale=self.scale,
        )

    @property
    def variables(self) -> Set[Variable]:
        return Variable.union(
            *(
                factor.factor.variables
                for factor in self.factors
                if factor.factor.variables is not None
            )
        )
