import abc
from collections import defaultdict
from typing import Dict, Generator, Iterable, List, Tuple

from ..utils import exc_for_token
from .operator import Operator
from .token import Token

# Cached property was introduced in Python 3.8 (we currently support 3.7)
try:
    from functools import cached_property
except ImportError:  # pragma: no cover
    from cached_property import cached_property  # type: ignore


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
    def operator_table(self) -> Dict[str, List[Operator]]:
        operator_table = defaultdict(list)
        for operator in self.operators:
            operator_table[operator.symbol].append(operator)
        for symbol in operator_table:
            operator_table[symbol] = sorted(
                operator_table[symbol],
                key=lambda op: (op.precedence, op.arity),
                reverse=True,
            )
        return operator_table

    def resolve(
        self, token: Token
    ) -> Generator[Tuple[Token, Iterable[Operator]], None, None]:
        """
        Generate the sets of operator candidates that may be viable for the
        given token (which may include multiple adjacent operators concatenated
        together). Each item generated must be a tuple for the token associated
        with the operator, and an iterable of `Operator` instances which should
        be considered by the AST generator. These `Operator` instances *MUST* be
        sorted in descending order of precendence and arity.

        Args:
            token: The operator `Token` instance for which `Operator`(s) should
                be resolved.
            max_prefix_arity: The number operator unclaimed tokens preceding the
                operator in the formula string.
            context: The current list of operators into which the operator to be
                resolved will be placed. This will be a list of `Operator`
                instances or tokens (tokens are return for grouping operators).
        """
        yield self._resolve(token, token.token)

    def _resolve(
        self,
        token: Token,
        symbol: str,
    ) -> Tuple[Token, Iterable[Operator]]:
        """
        The default operator resolving logic.
        """
        if symbol not in self.operator_table:
            raise exc_for_token(token, f"Unknown operator '{symbol}'.")
        return token, self.operator_table[symbol]

    # The operator table cache may not be pickleable, so let's drop it.
    def __getstate__(self) -> Dict:
        return {}
