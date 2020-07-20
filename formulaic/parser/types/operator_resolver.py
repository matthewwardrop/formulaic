from collections import defaultdict
from typing import List

from ..utils import exc_for_token
from .operator import Operator
from .token import Token


class OperatorResolver:

    def __init__(self):
        self.operator_table = defaultdict(list)
        for operator in self.operators:
            self.operator_table[operator.symbol].append(operator)
        for symbol in self.operator_table:
            self.operator_table[symbol] = sorted(self.operator_table[symbol], key=lambda op: op.precedence, reverse=True)

    @property
    def operators(self) -> List[Operator]:
        return []  # pragma: no cover

    def resolve(self, token: Token, max_prefix_arity) -> List[Operator]:
        return [self._resolve(token, token.token, max_prefix_arity)]

    def _resolve(self, token: Token, symbol: str, max_prefix_arity: int) -> Operator:
        if symbol not in self.operator_table:
            raise exc_for_token(token, f"Unknown operator '{symbol}'.")
        candidates = [
            candidate
            for candidate in self.operator_table[symbol]
            if max_prefix_arity == 0 and candidate.fixity is Operator.Fixity.PREFIX or max_prefix_arity > 0 and candidate.fixity is not Operator.Fixity.PREFIX
        ]
        if not candidates:
            raise exc_for_token(token, f"Operator `{symbol}` is incorrectly used.")
        if len(candidates) > 1:
            raise exc_for_token(token, f"Ambiguous operator `{symbol}`. This is not usually a user error. Please report this!")
        return candidates[0]
