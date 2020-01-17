from formulaic.parser.types import Factor


class EvaluatedFactor:

    def __init__(self, factor, values, kind=None, spans_intercept=False):
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
        if isinstance(other, EvaluatedFactor):
            return self.factor == other.factor
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, EvaluatedFactor):
            return self.factor < other.factor
        return NotImplemented
