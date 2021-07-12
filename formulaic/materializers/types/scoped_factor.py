class ScopedFactor:
    def __init__(self, factor, reduced=False):
        self.factor = factor
        self.reduced = reduced

    def __repr__(self):
        return repr(self.factor) + ("-" if self.reduced else "")

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, ScopedFactor):
            return self.factor == other.factor and self.reduced == other.reduced
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, ScopedFactor):
            if self.factor == other.factor:
                return self.reduced > other.reduced
            return self.factor < other.factor
        return NotImplemented
