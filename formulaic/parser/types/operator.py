from __future__ import annotations

from enum import Enum
from numbers import Number
from typing import Callable, List, Iterable, Union

from .term import Term
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
        precedence: Number,
        associativity: Union[str, Associativity] = "none",
        fixity: Union[str, Fixity] = "infix",
        to_terms: Callable[..., Iterable[Term]] = None,
        accepts_context: Callable[[List[Union[Token, Operator]]], bool] = None,
    ):
        self.symbol = symbol
        self.arity = arity
        self.precedence = precedence
        self.associativity = associativity
        self.fixity = fixity
        self._to_terms = to_terms
        self._accepts_context = accepts_context

    @property
    def associativity(self):
        return self._associativity

    @associativity.setter
    def associativity(self, associativity):
        self._associativity = Operator.Associativity(associativity or "none")

    @property
    def fixity(self):
        return self._fixity

    @fixity.setter
    def fixity(self, fixity):
        self._fixity = Operator.Fixity(fixity)

    def to_terms(self, *args):
        if self._to_terms is None:
            raise RuntimeError(f"`to_terms` is not implemented for '{self.symbol}'.")
        return self._to_terms(*args)

    def accepts_context(self, context: List[Union[Token, Operator]]):
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

    def __repr__(self):
        return self.symbol
