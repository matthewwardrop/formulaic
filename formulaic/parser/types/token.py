from enum import Enum

from .factor import Factor
from .term import Term


class Token:

    class Kind(Enum):
        OPERATOR = 'operator'
        NAME = 'name'
        PYTHON = 'python'
        VALUE = 'value'

    __slots__ = ('token', '_kind', 'source', 'source_start', 'source_end')

    def __init__(self, token='', *, kind=None, source_start=None, source_end=None, source=None):
        self.token = token
        self.kind = kind
        self.source = source
        self.source_start = source_start
        self.source_end = source_end

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = self.Kind(kind) if kind else kind

    def __bool__(self):
        return bool(self.token)

    def update(self, char, source_index, kind=None):
        self.token += char
        if self.source_start is None:
            self.source_start = source_index
        self.source_end = source_index
        if kind is not None:
            self.kind = kind
        return self

    def __eq__(self, other):
        if isinstance(other, str):
            return self.token == other
        if isinstance(other, Token):
            return self.token == other.token
        return NotImplemented

    def __hash__(self):
        return self.token.__hash__()

    def __lt__(self, other):
        if isinstance(other, Token):
            return self.token < other.token
        return NotImplemented

    @property
    def source_loc(self):
        return (self.source_start, self.source_end)

    def to_factor(self):
        return Factor(expr=self.token, kind=self.kind.value)

    def to_terms(self):
        return {Term([self.to_factor()])}

    def __repr__(self):
        return self.token
