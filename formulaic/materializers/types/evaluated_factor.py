from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from formulaic.parser.types import Factor

from .factor_values import FactorValues, FactorValuesMetadata


@dataclass
class EvaluatedFactor:
    """
    A container for the evaluated state of a `Factor` object in a given context.

    This class acts as the glue between an abstract `Factor` specification and
    the realisation of that factor in a specific data context.

    Attributes:
        factor: The `Factor` instance for which values have been computed.
        values: The evaluated values for the factor.
    """

    factor: Optional[Factor] = None
    values: Optional[FactorValues[Any]] = None

    @property
    def expr(self) -> str:
        """
        The expression of the evaluated factor.
        """
        return self.factor.expr

    @property
    def metadata(self) -> FactorValuesMetadata:
        """
        The metadata associated with the evaluated values.
        """
        return self.values.__formulaic_metadata__

    def __repr__(self) -> str:
        return repr(self.factor)

    def __eq__(self, other) -> bool:
        if isinstance(other, EvaluatedFactor):
            return self.factor == other.factor
        return NotImplemented

    def __lt__(self, other) -> bool:
        if isinstance(other, EvaluatedFactor):
            return self.factor < other.factor
        return NotImplemented

    def replace(self, **changes) -> EvaluatedFactor:
        """
        Return a copy of this `EvaluatedFactor` instance with the nominated
        attributes mutated.
        """
        return replace(self, **changes)
