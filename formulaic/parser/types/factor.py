from enum import Enum

from .term import Term


class Factor:

    class EvalMethod(Enum):
        LITERAL = 'literal'
        LOOKUP = 'lookup'
        PYTHON = 'python'

    __slots__ = ('expr', '_eval_method')

    def __init__(self, expr='', *, eval_method=None):
        self.expr = expr
        self.eval_method = eval_method

    @property
    def eval_method(self):
        return self._eval_method

    @eval_method.setter
    def eval_method(self, eval_method):
        self._eval_method = self.EvalMethod(eval_method) if eval_method else eval_method

    def __eq__(self, other):
        if isinstance(other, str):
            return self.expr == other
        if isinstance(other, Factor):
            return self.expr == other.expr
        return NotImplemented

    def __hash__(self):
        return self.expr.__hash__()

    def __lt__(self, other):
        if isinstance(other, Factor):
            return self.expr < other.expr
        return NotImplemented

    def to_terms(self):
        return {Term([self])}

    def __repr__(self):
        return self.expr
