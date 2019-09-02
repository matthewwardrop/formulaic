from enum import Enum


class EvaluatedFactor:

    class Kind(Enum):
        NUMERICAL = 'numerical'
        CATEGORICAL = 'categorical'
        CONSTANT = 'constant'

    def __init__(self, factor, values, kind='numerical'):
        self.factor = factor
        self.values = values
        self.kind = EvaluatedFactor.Kind(kind)

    @property
    def expr(self):
        return self.factor.expr

    def __repr__(self):
        return repr(self.factor)

    def __eq__(self, other):
        return self.factor == other.factor

    def __lt__(self, other):
        return self.factor < other.factor


class ScopedFactor:

    def __init__(self, factor, reduced=False):
        self.factor = factor
        self.reduced = reduced

    def __repr__(self):
        return repr(self.factor) + ('-' if self.reduced else '')

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, int):
            return False
        return self.factor == other.factor and self.reduced == other.reduced

    def __lt__(self, other):
        return self.factor < other.factor


class ScopedTerm:

    def __init__(self, factors, scale=None):
        self.factors = factors
        self.scale = scale

    def __hash__(self):
        return hash(self.factors)

    def __eq__(self, other):
        if isinstance(other, ScopedTerm):
            return self.factors == other.factors
        return NotImplemented
