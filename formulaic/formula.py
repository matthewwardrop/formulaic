import inspect

from .errors import FormulaInvalidError, FormulaMaterializerInvalidError
from .materializers.base import FormulaMaterializer
from .parser import FormulaParser
from .parser.types import Structured, Term
from .utils.calculus import differentiate_term


class Formula:
    @classmethod
    def from_spec(cls, spec, parser=None):
        if isinstance(spec, Formula):
            return spec
        return cls(spec, parser=parser)

    def __init__(self, formula, parser=None):
        parser = parser or FormulaParser()
        if isinstance(formula, str):
            terms = parser.get_terms(formula, sort=True)
        elif isinstance(formula, (list, set)):
            terms = [
                term
                for value in formula
                for term in (
                    parser.get_terms(value, include_intercept=False)
                    if isinstance(value, str)
                    else [value]
                )
            ]
        elif isinstance(formula, tuple):
            terms = tuple(
                Formula.from_spec(group, parser=parser).terms for group in formula
            )
        else:
            raise FormulaInvalidError(
                f"Unrecognized formula specification: {repr(formula)}."
            )
        self.terms = terms

    @property
    def terms(self):
        return self._terms

    @terms.setter
    def terms(self, terms):
        self.__check_terms(terms)
        self._terms = terms

    @classmethod
    def __check_terms(cls, terms, depth=0):
        if isinstance(terms, Structured):
            terms._map(cls.__check_terms)
        elif depth == 0 and isinstance(terms, tuple):
            for termset in terms:
                cls.__check_terms(termset, depth=depth + 1)
        else:
            for term in terms:
                if not isinstance(term, Term):
                    raise FormulaInvalidError(
                        f"All terms in formula should be instances of `formulaic.parser.types.Term`; received term {repr(term)} of type `{type(term)}`."
                    )

    def get_model_matrix(
        self, data, context=None, materializer=None, ensure_full_rank=True, **kwargs
    ):
        if materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(materializer)
        if not inspect.isclass(materializer) or not issubclass(
            materializer, FormulaMaterializer
        ):
            raise FormulaMaterializerInvalidError(
                "Materializers must be subclasses of `formulaic.materializers.FormulaMaterializer`."
            )
        return materializer(data, context=context or {}).get_model_matrix(
            self, ensure_full_rank=ensure_full_rank, **kwargs
        )

    def differentiate(self, *vars, use_sympy=False):
        return Formula(
            [differentiate_term(term, vars, use_sympy=use_sympy) for term in self.terms]
        )

    def __getattr__(self, attr):
        if attr in ("__getstate__", "__setstate__"):
            raise AttributeError(attr)
        if isinstance(self.terms, Structured) and attr in self.terms._to_dict(
            recurse=False
        ):
            return Formula(self.terms[attr])
        raise AttributeError(f"This formula has no substructures keyed by '{attr}'.")

    def __getitem__(self, item):
        if (
            isinstance(self.terms, tuple)
            or isinstance(self.terms, Structured)
            and self.terms._has_root
            and isinstance(self.terms.root, tuple)
        ):
            return Formula(self.terms[item])
        raise KeyError(
            f"This formula does not have any sub-parts indexable via `{repr(item)}`."
        )

    def __str__(self):
        terms = self.terms
        if isinstance(self.terms, tuple):
            terms = Structured(terms)
        if isinstance(terms, Structured):
            return str(
                terms._map(lambda terms: " + ".join(str(term) for term in terms))
            )
        return " + ".join(str(term) for term in terms)

    def __eq__(self, other):
        if isinstance(other, Formula):
            return self.terms == other.terms
        return NotImplemented

    def __repr__(self):
        return str(self)
