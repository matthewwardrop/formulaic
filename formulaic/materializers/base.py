import functools
import itertools
import operator
from abc import abstractmethod

from interface_meta import InterfaceMeta, quirk_docs

from formulaic.model_matrix import ModelMatrix
from formulaic.utils.context import LayeredContext
from formulaic.utils.stateful_transforms import stateful_eval

from ._transforms import TRANSFORMS
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
    def __init__(self, data, context=None, **kwargs):
        self.data = data
        self.context = context or {}
        self.factor_cache = {}
        self._init(**kwargs)

        self.layered_context = LayeredContext(self.data_context, self.context, TRANSFORMS)

    def _init(self):
        pass

    @property
    def data_context(self):
        return self.data

    @property
    def nrows(self):
        return len(self.data)

    def get_model_matrix(self, spec, ensure_full_rank=True):
        from formulaic.formula import Formula
        from formulaic.model_spec import ModelSpec

        if isinstance(spec, Formula):
            spec = ModelSpec(formula=spec, materializer=self, ensure_full_rank=ensure_full_rank)
        if not isinstance(spec, ModelSpec):
            raise RuntimeError("`spec` must be a `ModelSpec` or `Formula` instance.")

        # Step 1: Evaluate all factors
        for term in spec.formula.terms:
            for factor in term.factors:
                self._evaluate_factor(factor, spec.transforms, spec.encoding)

        # Step 2: Determine strategy to maintain structural full-rankness of output matrix
        scoped_terms = self._get_scoped_terms(spec.formula.terms, ensure_full_rank=spec.ensure_full_rank)

        # Step 3: Generate the columns which will be collated into the full matrix
        cols = {}
        for scoped_term in scoped_terms:
            if not scoped_term.factors:
                cols['Intercept'] = self._encode_constant(1, spec.encoding)
            else:
                cols.update(
                    self._get_columns_for_factors([
                        self._encode_evaled_factor(scoped_factor.factor, spec.encoding, reduced_rank=scoped_factor.reduced)
                        for scoped_factor in sorted(scoped_term.factors)
                    ], scale=scoped_term.scale)
                )

        # Step 4: Populate remaining model spec fields
        spec.feature_names = list(cols)
        spec.materializer = self

        # Step 5: Collate factors into one ModelMatrix
        return ModelMatrix(self._combine_columns(cols), spec=spec)

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
            transform_state (dict): The state of any stateful transforms
                (will be populated if empty).

        Returns:
            list<ScopedTerm>: A list of appropriately scoped terms.
        """
        scoped_terms = []
        spanned = set()

        for term in terms:
            evaled_factors = [
                self.factor_cache[factor.expr]
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

    def _evaluate_factor(self, factor, transform_state, encoder_state):
        if factor.expr not in self.factor_cache:
            # TODO: Cache identical evaluations
            if factor.kind.value == 'name':
                value = self._lookup(factor.expr)
            elif factor.kind.value == 'python':
                value = self._evaluate(factor.expr, transform_state)
            elif factor.kind.value == 'value':
                value = EvaluatedFactor(factor, self._evaluate(factor.expr, transform_state), kind='constant')
            else:
                raise ValueError(factor)

            if not isinstance(value, EvaluatedFactor):
                if factor.expr in encoder_state:
                    kind = encoder_state[factor.expr][0]
                elif self._is_categorical(value):
                    kind = 'categorical'
                else:
                    kind = 'numerical'
                value = EvaluatedFactor(
                    factor=factor,
                    values=value,
                    kind=kind
                )
            self.factor_cache[factor.expr] = value
        return self.factor_cache[factor.expr]

    def _lookup(self, name):
        return self.layered_context[name]

    def _evaluate(self, expr, transform_state):
        return stateful_eval(expr, self.layered_context, transform_state)

    @abstractmethod
    def _is_categorical(self, values):
        pass

    def _encode_evaled_factor(self, factor, encoder_state, reduced_rank=False):
        state = encoder_state.get(factor.expr, [None, {}])[1]
        if factor.kind.value == 'categorical':
            encoded = self._encode_categorical(factor.values, state, reduced_rank=reduced_rank)
        elif factor.kind.value == 'numerical':
            encoded = self._encode_numerical(factor.values, state)
        elif factor.kind.value == 'constant':
            encoded = self._encode_constant(factor.values, state)
        else:
            raise RuntimeError()

        encoder_state[factor.expr] = (factor.kind, state)

        if isinstance(encoded, dict):
            return {
                f'{factor.expr}[{category}]': values
                for category, values in encoded.items()
            }
        return {factor.expr: encoded}

    @abstractmethod
    def _encode_constant(self, value, encoder_state):
        pass

    @abstractmethod
    def _encode_categorical(self, values, encoder_state, reduced_rank=False):
        pass

    @abstractmethod
    def _encode_numerical(self, values, encoder_state):
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
