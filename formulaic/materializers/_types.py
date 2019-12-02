from formulaic.parser.types import Factor


class EvaluatedFactor:

    def __init__(self, factor, values, kind='numerical', spans_intercept=False):
        self.factor = factor
        self.values = values
        self.kind = kind
        self.spans_intercept = spans_intercept

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        if not kind or kind == 'unknown':
            raise ValueError("`EvaluatedFactor` instances must have a known kind.")
        self._kind = Factor.Kind(kind)

    @property
    def expr(self):
        return self.factor.expr

    @property
    def metadata(self):
        return self.factor.metadata

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
        self.factors = tuple(sorted(factors))
        self.scale = scale

    def __hash__(self):
        return hash(self.factors)

    def __eq__(self, other):
        if isinstance(other, ScopedTerm):
            return self.factors == other.factors
        return NotImplemented

    def __repr__(self):
        if not self.factors:
            return "1"
        return ":".join(f.__repr__() for f in sorted(self.factors))

    def copy(self):
        return ScopedTerm(tuple(self.factors), scale=self.scale)
