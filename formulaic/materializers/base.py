import functools
import itertools
import operator
from abc import abstractmethod
from collections import OrderedDict

from interface_meta import InterfaceMeta, quirk_docs

from formulaic.errors import FactorEncodingError, FactorEvaluationError, FormulaMaterializerNotFoundError
from formulaic.model_matrix import ModelMatrix
from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.stateful_transforms import stateful_eval

from formulaic.parser.types import Factor

from ._transforms import TRANSFORMS
from ._types import EvaluatedFactor, ScopedFactor, ScopedTerm


class FormulaMaterializer(metaclass=InterfaceMeta):

    REGISTRY = {}
    DEFAULTS = {}

    REGISTRY_NAME = None
    DEFAULT_FOR = None

    class Config:
        __slots__ = ('bespoke', 'sparse')

        def __init__(self, sparse=False, **bespoke):
            self.sparse = sparse
            self.bespoke = bespoke

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
            if materializer not in cls.REGISTRY:
                raise FormulaMaterializerNotFoundError(materializer)
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
        self._init(kwargs)
        self.config = self.Config(**kwargs)

        self.layered_context = LayeredMapping(self.data_context, self.context, TRANSFORMS)

        self.factor_cache = {}
        self.encoded_cache = {}

    def _init(self, kwargs):
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
            spec = ModelSpec(formula=Formula(spec), materializer=self, ensure_full_rank=ensure_full_rank)

        # Step 0: Check whether formula separators are in play, and if so, recurse.
        if isinstance(spec.formula.terms, tuple):
            return tuple(
                self.get_model_matrix(Formula(terms), ensure_full_rank=ensure_full_rank)
                for terms in spec.formula.terms
            )

        # Step 1: Evaluate all factors
        for term in spec.formula.terms:
            for factor in term.factors:
                self._evaluate_factor(factor, spec.transforms, spec.encoding)

        # Step 2: Determine strategy to maintain structural full-rankness of output matrix
        scoped_terms_for_terms = self._get_scoped_terms(spec.formula.terms, ensure_full_rank=spec.ensure_full_rank)

        # Step 3: Generate the columns which will be collated into the full matrix
        cols = []
        for term, scoped_terms in scoped_terms_for_terms:
            scoped_cols = OrderedDict()
            for scoped_term in scoped_terms:
                if not scoped_term.factors:
                    scoped_cols['Intercept'] = scoped_term.scale * self._encode_constant(1, None, {})
                else:
                    scoped_cols.update(
                        self._get_columns_for_term([
                            self._encode_evaled_factor(scoped_factor.factor, spec.encoding, reduced_rank=scoped_factor.reduced)
                            for scoped_factor in sorted(scoped_term.factors)
                        ], scale=scoped_term.scale)
                    )
            cols.append(
                (term, scoped_terms, scoped_cols)
            )

        # Step 4: Populate remaining model spec fields
        spec.materializer = self
        if spec.structure:
            cols = self._enforce_structure(cols, spec.structure)
        else:
            spec.structure = [
                (term, scoped_terms, list(scoped_cols))
                for term, scoped_terms, scoped_cols in cols
            ]

        # Step 5: Collate factors into one ModelMatrix
        return ModelMatrix(
            self._combine_columns([
                (name, values)
                for term, scoped_terms, scoped_cols in cols
                for name, values in scoped_cols.items()
            ]),
            spec=spec
        )

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
        spanned = set()

        for term in terms:
            evaled_factors = [
                self.factor_cache[factor.expr]
                for factor in term.factors
            ]

            if ensure_full_rank:
                term_span = self._get_scoped_terms_spanned_by_evaled_factors(evaled_factors).difference(spanned)
                scoped_terms = self._simplify_scoped_terms(term_span)
                spanned.update(term_span)
            else:
                scoped_terms = [
                    ScopedTerm(
                        factors=(
                            ScopedFactor(evaled_factor, reduced=False)
                            for evaled_factor in evaled_factors
                            if evaled_factor.kind.value != 'constant'
                        ),
                        scale=functools.reduce(operator.mul, [
                            evaled_factor.values
                            for evaled_factor in evaled_factors
                            if evaled_factor.kind.value == 'constant'
                        ], 1)
                    )
                ]
            yield term, scoped_terms

    @classmethod
    def _get_scoped_terms_spanned_by_evaled_factors(cls, evaled_factors):
        """
        Return the set of ScopedTerm instances which span the set of
        evaluated factors.

        Args:
            evaled_factors (iterable<EvaluatedFactor>)
        """
        scale = 1
        factors = []
        for factor in evaled_factors:
            if factor.kind.value == 'constant':
                scale *= factor.values
            elif factor.spans_intercept:
                factors.append((1, ScopedFactor(factor, reduced=True)))
            else:
                factors.append((ScopedFactor(factor),))
        return set(
            ScopedTerm(factors=(p for p in prod if p != 1), scale=scale)
            for prod in itertools.product(*factors)
        )

    @classmethod
    def _simplify_scoped_terms(cls, scoped_terms):
        """
        Return the minimal set of ScopedTerm instances that spans the same vectorspace.
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
                    terms = cls._simplify_scoped_terms(terms)
                    combined = True
                    break
            if not combined:
                terms.append(scoped_term.copy())
        return terms

    # Methods related to looking-up, evaluating and encoding terms and factors

    def _evaluate_factor(self, factor, transform_state, encoder_state):
        if factor.expr not in self.factor_cache:
            if factor.eval_method.value == 'lookup':
                value = self._lookup(factor.expr)
            elif factor.eval_method.value == 'python':
                value = self._evaluate(factor.expr, factor.metadata, transform_state)
            elif factor.eval_method.value == 'literal':
                value = EvaluatedFactor(factor, self._evaluate(factor.expr, factor.metadata, transform_state), kind='constant')
            else:
                raise FactorEvaluationError(f"Evaluation method {factor.eval_method.value} not recognised for factor {factor.expr}.")

            if not isinstance(value, EvaluatedFactor):
                if isinstance(value, dict) and '__kind__' in value:
                    kind = value['__kind__']
                    spans_intercept = value.get('__spans_intercept__', False)
                elif self._is_categorical(value):
                    kind = 'categorical'
                    spans_intercept = True
                else:
                    kind = 'numerical'
                    spans_intercept = False
                if factor.kind is not Factor.Kind.UNKNOWN and factor.kind.value != kind:
                    if factor.kind.value == 'categorical':
                        kind = factor.kind.value
                    else:
                        raise FactorEncodingError(f"Factor is expecting to be of kind '{factor.kind.value}' but is actually of kind '{kind}'.")
                if factor.expr in encoder_state and Factor.Kind(kind) is not encoder_state[factor.expr][0]:
                    raise FactorEncodingError(f"Factor kind `{kind}` does not match model specification of `{encoder_state[factor.expr][0]}`.")
                value = EvaluatedFactor(
                    factor=factor,
                    values=value,
                    kind=kind,
                    spans_intercept=spans_intercept,
                )
            self.factor_cache[factor.expr] = value
        return self.factor_cache[factor.expr]

    def _lookup(self, name):
        return self.layered_context[name]

    def _evaluate(self, expr, metadata, transform_state):
        return stateful_eval(expr, self.layered_context, {expr: metadata}, transform_state, self.config)

    def _is_categorical(self, values):
        if isinstance(values, dict):
            return values.get('__spans_intercept__', False)
        return False

    def _encode_evaled_factor(self, factor, encoder_state, reduced_rank=False):
        if not isinstance(factor.values, dict) or not factor.values.get('__encoded__', False):
            if factor.expr in self.encoded_cache:
                encoded = self.encoded_cache[factor.expr]
            elif (factor.expr, reduced_rank) in self.encoded_cache:
                encoded = self.encoded_cache[(factor.expr, reduced_rank)]
            else:
                state = encoder_state.get(factor.expr, [None, {}])[1]
                if factor.kind.value == 'categorical':
                    encoded = self._encode_categorical(factor.values, factor.metadata, state, reduced_rank=reduced_rank)
                elif factor.kind.value == 'numerical':
                    encoded = self._encode_numerical(factor.values, factor.metadata, state)
                elif factor.kind.value == 'constant':
                    encoded = self._encode_constant(factor.values, factor.metadata, state)
                else:
                    raise FactorEncodingError(factor)
                encoder_state[factor.expr] = (factor.kind, state)

                if isinstance(encoded, dict) and encoded.get('__drop_field__'):
                    cache_key = factor.expr
                else:
                    cache_key = (factor.expr, reduced_rank)

                self.encoded_cache[cache_key] = encoded
        else:
            encoded = factor.values

        # Encoded factors will now all be dicts
        if isinstance(encoded, dict) and encoded.get('__spans_intercept__') and reduced_rank:
            assert '__drop_field__' in encoded
            encoded = encoded.copy()
            del encoded[encoded['__drop_field__']]

        return self._flatten_encoded_evaled_factor(factor.expr, encoded)

    def _flatten_encoded_evaled_factor(self, name, values):
        if not isinstance(values, dict):
            return {name: values}

        name_format = values.get('__format__', '{name}[{field}]')

        flattened = {}
        for subfield, value in values.items():
            if isinstance(subfield, str) and subfield.startswith('__'):
                continue
            subname = name_format.format(name=name, field=subfield)
            if isinstance(value, dict):
                flattened.update(self._flatten_encoded_evaled_factor(subname, value))
            else:
                flattened[subname] = value

        return flattened

    @abstractmethod
    def _encode_constant(self, value, metadata, encoder_state):
        pass

    @abstractmethod
    def _encode_categorical(self, values, metadata, encoder_state, reduced_rank=False):
        pass

    @abstractmethod
    def _encode_numerical(self, values, metadata, encoder_state):
        pass

    # Methods related to ModelMatrix output

    def _enforce_structure(self, cols, structure):
        assert len(cols) == len(structure)
        for i in range(len(cols)):
            scoped_cols = cols[i][2]
            target_cols = structure[i][2]
            if len(scoped_cols) < len(target_cols):
                if len(scoped_cols) == 0:
                    col = self._encode_constant(0, None, None)
                elif len(scoped_cols) == 1:
                    col = next(scoped_cols.values())
                else:
                    raise FactorEncodingError(f"Structure of columns for term `{cols[i][0]}` inconsistent with specification: generated {list(scoped_cols)}, expecting {target_cols}.")
                scoped_cols = {
                    name: col
                    for name in target_cols
                }
            else:
                assert list(scoped_cols) == target_cols
            yield cols[i][0], cols[i][1], scoped_cols

    def _get_columns_for_term(self, factors, scale=1):
        """
        Assemble the columns for a model matrix given factors and a scale.

        This performs the row-wise Kronecker product of the factors.

        Args:
            factors
            scale

        Returns:
            dict
        """
        out = OrderedDict()
        for product in itertools.product(*(factor.items() for factor in factors)):
            out[':'.join(p[0] for p in product)] = scale * functools.reduce(operator.mul, (p[1] for p in product))
        return out

    @abstractmethod
    def _combine_columns(self, cols):
        pass
