from enum import Enum


class Operator:

    class Associativity(Enum):
        LEFT = 'left'
        RIGHT = 'right'
        NONE = 'none'

    class Fixity(Enum):
        PREFIX = 'prefix'
        INFIX = 'infix'
        POSTFIX = 'postfix'

    def __init__(self, symbol, *, arity=None, precedence=None, associativity=None, fixity='infix', to_terms=None):
        self.symbol = symbol
        self.arity = arity
        self.precedence = precedence
        self.associativity = associativity
        self.fixity = fixity
        self._to_terms = to_terms

    @property
    def associativity(self):
        return self._associativity

    @associativity.setter
    def associativity(self, associativity):
        self._associativity = Operator.Associativity(associativity or 'none')

    @property
    def fixity(self):
        return self._fixity

    @fixity.setter
    def fixity(self, fixity):
        self._fixity = Operator.Fixity(fixity)

    def to_terms(self, *args):
        if self._to_terms is None:
            raise RuntimeError(f"`to_terms` is not implemented for '{self.symbol}'.")
        return self._to_terms(*args)

    def __repr__(self):
        return self.symbol
