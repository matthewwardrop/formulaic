from enum import Enum

from .term import Term


class Factor:

    class EvalMethod(Enum):
        UNKNOWN = 'unknown'
        LITERAL = 'literal'
        LOOKUP = 'lookup'
        PYTHON = 'python'

    class Kind(Enum):
        UNKNOWN = 'unknown'
        CONSTANT = 'constant'
        NUMERICAL = 'numerical'
        CATEGORICAL = 'categorical'

    __slots__ = ('expr', '_eval_method', '_kind', 'metadata')

    def __init__(self, expr='', *, eval_method=None, kind=None, metadata=None):
        self.expr = expr
        self.eval_method = eval_method
        self.kind = kind
        self.metadata = metadata or {}

    @property
    def eval_method(self):
        return self._eval_method

    @eval_method.setter
    def eval_method(self, eval_method):
        self._eval_method = Factor.EvalMethod(eval_method or 'unknown')

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = Factor.Kind(kind or 'unknown')

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
