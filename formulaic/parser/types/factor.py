from enum import Enum

from .term import Term


class Factor:

    class Kind(Enum):
        NAME = 'name'
        VALUE = 'value'
        PYTHON = 'python'

    __slots__ = ('expr', '_kind')

    def __init__(self, expr='', *, kind=None):
        self.expr = expr
        self.kind = kind

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = self.Kind(kind) if kind else kind

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
