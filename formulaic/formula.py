import inspect

from .errors import FormulaInvalidError, FormulaMaterializerInvalidError
from .materializers.base import FormulaMaterializer
from .parser import FormulaParser
from .parser.types import Term


class Formula:

    def __init__(self, formula, parser=None):
        if isinstance(formula, str):
            terms = (parser or FormulaParser())(formula)
        elif isinstance(formula, (tuple, list, set)):
            terms = formula
        else:
            raise FormulaInvalidError(f"Unrecognized formula specification: {repr(formula)}.")
        self.terms = terms

    @property
    def terms(self):
        return self._terms

    @terms.setter
    def terms(self, terms):
        self.__check_terms(terms)
        self._terms = self.__sort_terms(terms)

    @classmethod
    def __check_terms(cls, terms, depth=0):
        if depth == 0 and isinstance(terms, tuple):
            for termset in terms:
                cls.__check_terms(termset, depth=depth + 1)
        else:
            for term in terms:
                if not isinstance(term, Term):
                    raise FormulaInvalidError(f"All terms in formula should be instances of `formulaic.parser.types.Term`; received term {repr(term)} of type `{type(term)}`.")

    @classmethod
    def __sort_terms(cls, terms):
        if isinstance(terms, tuple):
            return tuple(sorted(ts) for ts in terms)
        else:
            return sorted(terms)

    def get_model_matrix(self, data, context=None, materializer=None, ensure_full_rank=True, **kwargs):
        if materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(materializer)
        if not inspect.isclass(materializer) or not issubclass(materializer, FormulaMaterializer):
            raise FormulaMaterializerInvalidError("Materializers must be subclasses of `formulaic.materializers.FormulaMaterializer`.")
        return materializer(data, context=context or {}, **kwargs).get_model_matrix(self, ensure_full_rank=ensure_full_rank)

    def __str__(self):
        if isinstance(self.terms, tuple):
            return ' ~ '.join(" + ".join(str(term) for term in terms) for terms in self._terms)
        return " + ".join(str(term) for term in self.terms)

    def __repr__(self):
        return str(self)
