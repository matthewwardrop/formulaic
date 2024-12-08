from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Callable, List, Mapping, Optional, Union

from .token import Token


class Operator:
    """
    Specification for how an operator in a formula string should behave.

    Attributes:
        symbol: The operator for which the configuration applies.
        arity: The number of arguments that this operator consumes.
        precedence: How tightly this operator binds its arguments (the higher
            the number, the more tightly it binds). Operators with higher
            precedence will be evaluated first.
        associativity: One of 'left', 'right', or 'none'; indicating how
            operators of the same precedence should be evaluated in the absence
            of explicit grouping parentheses. If left associative, groups are
            formed from the left [e.g. a % b % c -> ((a % b) % c)]; and
            similarly for right.
        fixity: One of 'prefix', 'infix', or 'postfix'; indicating how the
            operator is positioned relative to its arguments. If 'prefix', the
            operator comes before its arguments; if 'infix', the operator comes
            between its arguments (and there must be exactly two of them); and
            if 'postfix', the operator comes after its arguments.
        to_terms: A callable that maps the arguments pass to the operator to
            an iterable of `Term` instances.
        accepts_context: A callable that will receive a list of Operator and
            Token instances that describe the context in which the operator
            would be applied if this callable returns `True`.
        structural: Whether this operator adds structure to the terms sets, in
            which case `Structured._merge` will not be used in the
            `ASTNode.to_terms()`, and the termsets will be directly passed to
            `Operator.to_terms()`.
        disabled: Whether this operator is disabled and should not be used. This
            is useful for restricting the set of formula that can be parsed in
            certain contexts.
    """

    class Associativity(Enum):
        LEFT = "left"
        RIGHT = "right"
        NONE = "none"

    class Fixity(Enum):
        PREFIX = "prefix"
        INFIX = "infix"
        POSTFIX = "postfix"

    def __init__(
        self,
        symbol: str,
        *,
        arity: int,
        precedence: float,
        associativity: Union[None, str, Associativity] = Associativity.NONE,
        fixity: Union[str, Fixity] = Fixity.INFIX,
        to_terms: Optional[Callable[..., Any]] = None,
        accepts_context: Optional[
            Callable[[List[Union[Token, Operator]]], bool]
        ] = None,
        structural: bool = False,
        disabled: bool = False,
    ):
        self.symbol = symbol
        self.arity = arity
        self.precedence = precedence
        self.associativity = associativity  # type: ignore
        self.fixity = fixity  # type: ignore
        self._to_terms = to_terms
        self._accepts_context = accepts_context
        self.structural = structural
        self.disabled = disabled

    @property
    def associativity(self) -> Operator.Associativity:
        return self._associativity

    @associativity.setter
    def associativity(self, associativity: Union[str, Operator.Associativity]) -> None:
        self._associativity = Operator.Associativity(associativity or "none")

    @property
    def fixity(self) -> Operator.Fixity:
        return self._fixity

    @fixity.setter
    def fixity(self, fixity: Union[str, Operator.Fixity]) -> None:
        self._fixity = Operator.Fixity(fixity)

    def to_terms(self, *args: Any, context: Optional[Mapping[str, Any]] = None) -> Any:
        if self._to_terms is None:
            raise RuntimeError(f"`to_terms` is not implemented for '{self.symbol}'.")
        if inspect.signature(self._to_terms).parameters.get("context"):
            return self._to_terms(*args, context=context or {})
        return self._to_terms(*args)

    def accepts_context(self, context: List[Union[Token, Operator]]) -> bool:
        if self._accepts_context:
            # We only need to pass on tokens and operators with precedence less
            # than or equal to ourselves, since all other operators will be
            # evaluated before us.
            return self._accepts_context(
                [
                    c
                    for c in context
                    if isinstance(c, Token) or c.precedence <= self.precedence
                ]
            )
        return True

    def __repr__(self) -> str:
        return self.symbol
