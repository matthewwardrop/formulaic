import abc
from collections import defaultdict
from typing import List, Union

from ..utils import exc_for_token
from .operator import Operator
from .token import Token

# Cached property was introduced in Python 3.8 (we currently support 3.7)
try:
    from functools import cached_property
except ImportError:  # pragma: no cover
    from cached_property import cached_property


class OperatorResolver(metaclass=abc.ABCMeta):
    """
    Resolves which `Operator` instance should be used for a given operator
    `Token`.

    This class should be subclassed and have `.operators` and/or `.resolve()`
    overridden in order to achieve the desired formula algebra.

    Note: most users will probably be interested in extending/subclassing
    `DefaultOperatorResolver`, which implements the default formula operator
    logic. You should subclass this class directly only if you want to start
    from scratch.

    Attributes:
        operator_table: A cache of the mapping from operator symbol to
            `Operator` instances implementing it.
    """

    @property
    @abc.abstractmethod
    def operators(self) -> List[Operator]:
        """
        The `Operator` instance pool which can be matched to tokens by
        `.resolve()`.
        """

    @cached_property
    def operator_table(self):
        operator_table = defaultdict(list)
        for operator in self.operators:
            operator_table[operator.symbol].append(operator)
        for symbol in operator_table:
            operator_table[symbol] = sorted(
                operator_table[symbol], key=lambda op: op.precedence, reverse=True
            )
        return operator_table

    def resolve(
        self, token: Token, max_prefix_arity: int, context: List[Union[Token, Operator]]
    ) -> List[Operator]:
        """
        Return a list of operators to apply for a given token in the AST
        generation.

        Args:
            token: The operator `Token` instance for which `Operator`(s) should
                be resolved.
            max_prefix_arity: The number operator unclaimed tokens preceding the
                operator in the formula string.
            context: The current list of operators into which the operator to be
                resolved will be placed. This will be a list of `Operator`
                instances or tokens (tokens are return for grouping operators).
        """
        return [self._resolve(token, token.token, max_prefix_arity, context)]

    def _resolve(
        self,
        token: Token,
        symbol: str,
        max_prefix_arity: int,
        context: List[Union[Token, Operator]],
    ) -> Operator:
        """
        The default operator resolving logic.
        """
        if symbol not in self.operator_table:
            raise exc_for_token(token, f"Unknown operator '{symbol}'.")
        candidates = [
            candidate
            for candidate in self.operator_table[symbol]
            if (
                max_prefix_arity == 0
                and candidate.fixity is Operator.Fixity.PREFIX
                or max_prefix_arity > 0
                and candidate.fixity is not Operator.Fixity.PREFIX
            )
            and candidate.accepts_context(context)
        ]
        if not candidates:
            raise exc_for_token(token, f"Operator `{symbol}` is incorrectly used.")
        if len(candidates) > 1:
            raise exc_for_token(
                token,
                f"Ambiguous operator `{symbol}`. This is not usually a user error. Please report this!",
            )
        return candidates[0]

    # The operator table cache may not be pickleable, so let's drop it.
    def __getstate__(self):
        return {}
