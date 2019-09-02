import functools
import itertools
import operator
from abc import abstractmethod

from interface_meta import InterfaceMeta, quirk_docs

from formulaic.model_matrix import ModelMatrix
from formulaic.utils.context import LayeredContext

from ._types import EvaluatedFactor, ScopedFactor, ScopedTerm


class FormulaMaterializer(metaclass=InterfaceMeta):

    REGISTRY = {}
    DEFAULTS = {}

    REGISTRY_NAME = None
    DEFAULT_FOR = None

    # Registry methods

    @classmethod
    def __register_implementation__(cls):
        if 'REGISTRY_NAME' in cls.__dict__ and cls.REGISTRY_NAME:
            cls.REGISTRY[cls.REGISTRY_NAME] = cls
            if 'DEFAULT_FOR' in cls.__dict__ and cls.DEFAULT_FOR:
                for default in cls.DEFAULT_FOR:
                    cls.DEFAULTS[default] = cls

    @classmethod
    def for_materializer(cls, materializer):
        if isinstance(materializer, str):
            materializer = cls.REGISTRY[materializer]
        return materializer

    @classmethod
    def for_data(cls, data):
        datacls = data.__class__
        return cls.DEFAULTS[f"{datacls.__module__}.{datacls.__qualname__}"]

    # Public API

    @quirk_docs(method='_init')
    def __init__(self, data, context, **kwargs):
        self.data = data
        self.context = context
        self._init(**kwargs)

        self.layered_context = LayeredContext(self.data_context, self.context)

    def _init(self):
        pass

    @property
    def data_context(self):
        return self.data

    @property
    def nrows(self):
        return len(self.data)

    def get_model_matrix(self, formula, ensure_full_rank=True):

        cols = {}

        scoped_terms = self._get_scoped_terms(formula.terms, ensure_full_rank=ensure_full_rank)

        # Step 2: Generate the columns which will be collated into the full matrix
        for scoped_term in scoped_terms:
            if not scoped_term.factors:
                cols['Intercept'] = self._encode_constant(1)
            else:
                cols.update(
                    self._get_columns_for_factors([
                        self._encode_evaled_factor(scoped_factor.factor, reduced_rank=scoped_factor.reduced)
                        for scoped_factor in sorted(scoped_term.factors)
                    ], scale=scoped_term.scale)
                )

        # Step 3: Collate factors into one ModelMatrix
        return ModelMatrix(formula, self._combine_columns(cols), feature_names=list(cols), materializer=self)

    # Methods related to ensuring out matrices are structurally full-rank

    def _get_scoped_terms(self, terms, ensure_full_rank=True):
        """
        Generate the terms to be used in the model matrix.

        This method first evaluates each factor in the context of the data
        (and environment), and then determines the correct "scope" (full vs.
        reduced rank) for each term. If `ensure_full_rank` is `True`, then the
        resulting terms when combined is guaranteed to be structurally full-rank.

        Args:
            terms (list<Term>): A list of term arguments (usually from a formula)
                object.
            ensure_full_rank (bool): Whether evaluated terms should be scoped
                to ensure that their combination will result in a full-rank
                matrix.

        Returns:
            list<ScopedTerm>: A list of appropriately scoped terms.
        """
        scoped_terms = []
        spanned = set()

        for term in terms:
            evaled_factors = [
                self._evaluate_factor(factor)
                for factor in term.factors
            ]

            if ensure_full_rank:
                term_span = self._get_scoped_terms_spanned_by_evaled_factors(evaled_factors).difference(spanned)
                scoped_terms.extend(self._simplify_scoped_terms(term_span))
                spanned.update(term_span)
            else:
                scoped_terms.append(ScopedTerm(
                    factors=tuple(
                        ScopedFactor(evaled_factor, reduced=False)
                        for evaled_factor in evaled_factors
                        if evaled_factor.kind.value != 'constant'
                    ),
                    scale=functools.reduce(operator.mul, [
                        evaled_factor.values
                        for evaled_factor in evaled_factors
                        if evaled_factor.kind.value == 'constant'
                    ], 1)
                ))

        return scoped_terms

    def _get_scoped_terms_spanned_by_evaled_factors(self, evaled_factors):
        """
        Return the set of ScopedTerm instances which span the set of
        evaluated factors.

        Args:
            evaled_factors (iterable<EvaluatedFactor>)
        """
        scale = 1
        factors = []
        for factor in evaled_factors:
            if factor.kind.value == 'categorical':
                factors.append((1, ScopedFactor(factor, reduced=True)))
            elif factor.kind.value == 'numerical':
                factors.append((ScopedFactor(factor),))
            elif factor.kind.value == 'constant':
                scale *= factor.values
            else:
                raise RuntimeError("Unknown factor type.")
        return set(
            ScopedTerm(factors=tuple(sorted(p for p in prod if p != 1)), scale=scale)
            for prod in itertools.product(*factors)
        )

    def _simplify_scoped_terms(self, scoped_terms):
        """
        Return the minimal set of ScopedTerm instances that spans the same vectorspace.

        Warning: This method mutates inplace some scoped_terms.
        """
        terms = []
        for scoped_term in sorted(scoped_terms, key=lambda x: len(x.factors)):
            factors = set(scoped_term.factors)
            combined = False
            for co_scoped_term in terms:
                cofactors = set(co_scoped_term.factors)
                factors_diff = factors.difference(cofactors)
                if len(factors) - 1 != len(cofactors) or len(factors_diff) != 1:
                    continue
                factor_new = next(iter(factors_diff))
                if factor_new.reduced:
                    co_scoped_term.factors += (ScopedFactor(factor_new.factor, reduced=False), )
                    terms = self._simplify_scoped_terms(scoped_terms)
                    combined = True
                    break
            if not combined:
                terms.append(scoped_term)
        return terms

    # Methods related to looking-up, evaluating and encoding terms and factors

    def _evaluate_factor(self, factor):
        if factor.kind.value == 'name':
            value = self._lookup(factor.expr)
        elif factor.kind.value == 'python':
            value = self._evaluate(factor.expr)
        elif factor.kind.value == 'value':
            value = EvaluatedFactor(factor, self._evaluate(factor.expr), kind='constant')
        else:
            raise ValueError(factor)

        if not isinstance(value, EvaluatedFactor):
            value = EvaluatedFactor(
                factor=factor,
                values=value,
                kind='categorical' if self._is_categorical(value) else 'numerical'
            )
        return value

    def _lookup(self, name):
        return self.layered_context[name]

    def _evaluate(self, expr):
        return eval(expr, {}, self.layered_context)

    @abstractmethod
    def _is_categorical(self, values):
        pass

    def _encode_evaled_factor(self, factor, reduced_rank=False):
        if factor.kind.value == 'categorical':
            encoded = self._encode_categorical(factor.values, reduced_rank=reduced_rank)
        elif factor.kind.value == 'numerical':
            encoded = self._encode_numerical(factor.values)
        elif factor.kind.value == 'constant':
            encoded = self._encode_constant(factor.values)
        else:
            raise RuntimeError()

        if isinstance(encoded, dict):
            return {
                f'{factor.expr}[{category}]': values
                for category, values in encoded.items()
            }
        return {factor.expr: encoded}

    @abstractmethod
    def _encode_constant(self, value):
        pass

    @abstractmethod
    def _encode_categorical(self, values, reduced_rank=False):
        pass

    @abstractmethod
    def _encode_numerical(self, values):
        pass

    # Methods related to ModelMatrix output

    def _get_columns_for_factors(self, factors, scale=1):
        """
        Assemble the columns for a model matrix given factors and a scale.

        This performs the row-wise Kronecker product of the factors.

        Args:
            factors
            scale

        Returns:
            dict
        """
        out = {}
        for product in itertools.product(*(factor.items() for factor in factors)):
            out[':'.join(p[0] for p in product)] = scale * functools.reduce(operator.mul, (p[1] for p in product))
        return out

    @abstractmethod
    def _combine_columns(self, cols):
        pass
