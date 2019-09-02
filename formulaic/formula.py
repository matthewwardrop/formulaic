from .materializers.base import FormulaMaterializer
from .parser import FormulaParser


class Formula:

    def __init__(self, formula, parser=None):
        self.formula = formula
        self.parser = parser or FormulaParser()
        self._terms = None

    @property
    def terms(self):
        if self._terms is None:
            terms = self.parser(self.formula)
            if isinstance(terms, tuple):
                self._terms = tuple(sorted(ts) for ts in terms)
            else:
                self._terms = sorted(terms)
        return self._terms

    def get_model_matrix(self, data, context=None, materializer=None, ensure_full_rank=True, **kwargs):
        if materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(materializer)
        assert issubclass(materializer, FormulaMaterializer)
        return materializer(data, context=context or {}, **kwargs).get_model_matrix(self, ensure_full_rank=ensure_full_rank)

    def __repr__(self):
        if isinstance(self.terms, tuple):
            return ' ~ '.join(" + ".join(str(term) for term in terms) for terms in self._terms)
        return " + ".join(str(term) for term in self.terms)
