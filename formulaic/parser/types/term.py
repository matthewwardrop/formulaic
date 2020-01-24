class Term:

    def __init__(self, factors):
        self.factors = set(factors)

    def __mul__(self, other):
        if isinstance(other, Term):
            return Term([*self.factors, *other.factors])
        return NotImplemented

    @property
    def _tuple(self):
        return tuple(factor.expr for factor in sorted(self.factors))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, Term):
            return self._tuple == other._tuple
        if isinstance(other, str):
            return repr(self) == other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Term):
            if len(self.factors) == len(other.factors):
                return sorted(self.factors) < sorted(other.factors)
            if len(self.factors) < len(other.factors):
                return True
            return False
        return NotImplemented

    def __repr__(self):
        return ':'.join(self._tuple)
