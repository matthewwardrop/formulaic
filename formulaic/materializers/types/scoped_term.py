class ScopedTerm:

    __slots__ = ('factors', 'scale')

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
        factor_repr = ":".join(f.__repr__() for f in sorted(self.factors)) if self.factors else "1"
        if self.scale is not None and self.scale != 1:
            return f"{self.scale}*{factor_repr}"
        return factor_repr

    def copy(self):
        return ScopedTerm(tuple(self.factors), scale=self.scale)
