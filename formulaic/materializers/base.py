from __future__ import annotations

import functools
import itertools
import operator
from abc import abstractmethod
from collections import defaultdict, OrderedDict
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Iterable,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from interface_meta import InterfaceMeta, inherit_docs

from formulaic.errors import (
    FactorEncodingError,
    FactorEvaluationError,
    FormulaMaterializationError,
    FormulaMaterializerNotFoundError,
)
from formulaic.materializers.types.factor_values import FactorValuesMetadata
from formulaic.model_matrix import ModelMatrix
from formulaic.parser.types import Factor, Structured, Term
from formulaic.transforms import TRANSFORMS
from formulaic.utils.cast import as_columns
from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.stateful_transforms import stateful_eval

from .types import EvaluatedFactor, FactorValues, ScopedFactor, ScopedTerm

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec  # pragma: no cover


class FormulaMaterializerMeta(InterfaceMeta):

    REGISTERED_NAMES = {}
    REGISTERED_INPUTS = defaultdict(list)

    def __register_implementation__(cls):
        if "REGISTER_NAME" in cls.__dict__ and cls.REGISTER_NAME:
            cls.REGISTERED_NAMES[cls.REGISTER_NAME] = cls

            if "REGISTER_INPUTS" in cls.__dict__:
                for input_type in cls.REGISTER_INPUTS:
                    cls.REGISTERED_INPUTS[input_type] = sorted(
                        cls.REGISTERED_INPUTS[input_type] + [cls],
                        key=lambda x: x.REGISTER_PRECEDENCE,
                        reverse=True,
                    )

    def for_materializer(cls, materializer):
        if isinstance(materializer, str):
            if materializer not in cls.REGISTERED_NAMES:
                raise FormulaMaterializerNotFoundError(materializer)
            materializer = cls.REGISTERED_NAMES[materializer]
        return materializer

    def for_data(cls, data, output=None):
        datacls = data.__class__
        input_type = f"{datacls.__module__}.{datacls.__qualname__}"

        if input_type not in cls.REGISTERED_INPUTS:
            raise FormulaMaterializerNotFoundError(
                f"No materializer has been registered for input type {repr(input_type)}. Available input types are: {set(cls.REGISTER_INPUTS)}."
            )

        if output is None:
            return cls.REGISTERED_INPUTS[input_type][0]

        for materializer in cls.REGISTERED_INPUTS[input_type]:
            if output in materializer.REGISTER_OUTPUTS:
                return materializer

        output_types = set(
            *itertools.chain(
                materializer.REGISTER_OUTPUTS
                for materializer in cls.REGISTERED_INPUTS[input_type]
            )
        )
        raise FormulaMaterializerNotFoundError(
            f"No materializer has been registered for input type {repr(input_type)} that supports output type {repr(output)}. Available output types for {repr(input_type)} are: {output_types}."
        )


class FormulaMaterializer(metaclass=FormulaMaterializerMeta):

    REGISTER_NAME = None
    REGISTER_INPUTS = set()
    REGISTER_OUTPUTS = set()
    REGISTER_PRECEDENCE = 100

    # Public API

    @inherit_docs(method="_init")
    def __init__(self, data, context=None, **kwargs):
        self.data = data
        self.context = context or {}
        self._init(**kwargs)

        self.layered_context = LayeredMapping(
            self.data_context, self.context, TRANSFORMS
        )

        self.factor_cache = {}
        self.encoded_cache = {}

    def _init(self):
        pass  # pragma: no cover

    @property
    def data_context(self):
        return self.data

    @property
    def nrows(self):
        return len(self.data)

    def get_model_matrix(
        self, spec, ensure_full_rank=True, na_action="drop", output=None
    ):
        # Prepare arguments
        if output is None:
            output = self.REGISTER_OUTPUTS[0]
        if output not in self.REGISTER_OUTPUTS:
            raise FormulaMaterializationError(
                f"Nominated output {repr(output)} is invalid. Available output types are: {set(self.REGISTER_OUTPUTS)}."
            )

        # Prepare (potentially structured) ModelSpec(s)
        model_specs = self._prepare_model_specs(
            spec, ensure_full_rank=ensure_full_rank, na_action=na_action, output=output
        )
        if not isinstance(model_specs, Structured):
            model_specs = Structured(model_specs)

        # Step 0: Pool all factors and transform state, ensuring consistency
        # during factor evaluation (esp. which rows get dropped).
        (
            factors,
            factor_evaluation_model_spec,
        ) = self._prepare_factor_evaluation_model_spec(model_specs)

        # Step 1: Evaluate all factors and cache the results, keeping track of
        # which rows need dropping (if `self.config.na_action == 'drop'`).
        drop_rows = set()
        for factor in factors:
            self._evaluate_factor(factor, factor_evaluation_model_spec, drop_rows)
        drop_rows = sorted(drop_rows)

        # Step 2: Update the structured model specs with the information from
        # the shared transform state pool.
        model_specs._map(
            lambda ms: ms.transform_state.update(
                {
                    factor.expr: factor_evaluation_model_spec.transform_state[
                        factor.expr
                    ]
                    for term in ms.formula.terms
                    for factor in term.factors
                    if factor.expr in factor_evaluation_model_spec.transform_state
                }
            )
        )

        # Step 3: Build the model matrices using the shared factor cache, and
        # by recursing over the structured model matrices.
        model_matrices = model_specs._map(
            lambda model_spec: self._build_model_matrix(model_spec, drop_rows=drop_rows)
        )
        model_matrices._mapped_attrs = {"model_spec"}

        if len(model_matrices) == 1 and model_matrices._has_root:
            return model_matrices.root
        return model_matrices

    def _build_model_matrix(self, spec: ModelSpec, drop_rows):

        # Step 1: Determine strategy to maintain structural full-rankness of output matrix
        scoped_terms_for_terms = self._get_scoped_terms(
            spec.formula.terms, ensure_full_rank=spec.ensure_full_rank
        )

        # Step 2: Generate the columns which will be collated into the full matrix
        cols = []
        for term, scoped_terms in scoped_terms_for_terms:
            scoped_cols = OrderedDict()
            for scoped_term in scoped_terms:
                if not scoped_term.factors:
                    scoped_cols[
                        "Intercept"
                    ] = scoped_term.scale * self._encode_constant(
                        1, None, {}, spec, drop_rows
                    )
                else:
                    scoped_cols.update(
                        self._get_columns_for_term(
                            [
                                self._encode_evaled_factor(
                                    scoped_factor.factor,
                                    spec,
                                    drop_rows,
                                    reduced_rank=scoped_factor.reduced,
                                )
                                for scoped_factor in sorted(scoped_term.factors)
                            ],
                            spec=spec,
                            scale=scoped_term.scale,
                        )
                    )
            cols.append((term, scoped_terms, scoped_cols))

        # Step 3: Populate remaining model spec fields
        spec.materializer = self
        if spec.structure:
            cols = self._enforce_structure(cols, spec, drop_rows)
        else:
            spec.structure = [
                (
                    term,
                    list(st.copy(without_values=True) for st in scoped_terms),
                    list(scoped_cols),
                )
                for term, scoped_terms, scoped_cols in cols
            ]

        # Step 4: Collate factors into one ModelMatrix
        return ModelMatrix(
            self._combine_columns(
                [
                    (name, values)
                    for term, scoped_terms, scoped_cols in cols
                    for name, values in scoped_cols.items()
                ],
                spec=spec,
                drop_rows=drop_rows,
            ),
            spec=spec,
        )

    # Methods related to input preparation

    def _prepare_model_specs(
        self,
        spec,
        ensure_full_rank,
        na_action,
        output,
    ) -> Union[ModelSpec, Structured[ModelSpec]]:
        from formulaic.formula import Formula
        from formulaic.model_spec import ModelSpec

        if isinstance(spec, Structured):
            return spec._map(
                functools.partial(
                    self._prepare_model_specs,
                    ensure_full_rank=ensure_full_rank,
                    na_action=na_action,
                    output=output,
                ),
                recurse=False,
            )
        elif isinstance(spec, ModelSpec):
            return spec

        formula = Formula.from_spec(spec)
        if isinstance(formula.terms, Structured):
            return self._prepare_model_specs(
                formula.terms,
                ensure_full_rank=ensure_full_rank,
                na_action=na_action,
                output=output,
            )
        return ModelSpec(
            formula=formula,
            materializer=self,
            ensure_full_rank=ensure_full_rank,
            na_action=na_action,
            output=output,
        )

    def _prepare_factor_evaluation_model_spec(self, model_specs: Structured[ModelSpec]):
        from formulaic.model_spec import ModelSpec

        output = set()
        na_action = set()
        ensure_full_rank = set()
        factors = set()
        transform_state = {}

        def update_pooled_spec(model_spec: ModelSpec):
            output.add(model_spec.output)
            na_action.add(model_spec.na_action)
            ensure_full_rank.add(model_spec.ensure_full_rank)
            factors.update(
                itertools.chain(*(term.factors for term in model_spec.formula.terms))
            )
            transform_state.update(
                model_spec.transform_state
            )  # TODO: Check for consistency?

        model_specs._map(update_pooled_spec)

        if len(output) != 1 or len(na_action) != 1 or len(ensure_full_rank) != 1:
            raise RuntimeError(
                "Provided `ModelSpec` instances are not consistent."
            )  # pragma: no cover; will only occur if users manually construct a structured model spec.

        return factors, ModelSpec(
            formula=[],
            ensure_full_rank=next(iter(ensure_full_rank)),
            na_action=next(iter(na_action)),
            output=next(iter(output)),
            transform_state=transform_state,
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
            evaled_factors = [self.factor_cache[factor.expr] for factor in term.factors]

            if ensure_full_rank:
                term_span = self._get_scoped_terms_spanned_by_evaled_factors(
                    evaled_factors
                ).difference(spanned)
                scoped_terms = self._simplify_scoped_terms(term_span)
                spanned.update(term_span)
            else:
                scoped_terms = [
                    ScopedTerm(
                        factors=(
                            ScopedFactor(evaled_factor, reduced=False)
                            for evaled_factor in evaled_factors
                            if evaled_factor.metadata.kind is not Factor.Kind.CONSTANT
                        ),
                        scale=functools.reduce(
                            operator.mul,
                            [
                                evaled_factor.values
                                for evaled_factor in evaled_factors
                                if evaled_factor.metadata.kind.value
                                is Factor.Kind.CONSTANT
                            ],
                            1,
                        ),
                    )
                ]
            yield term, scoped_terms

    @classmethod
    def _get_scoped_terms_spanned_by_evaled_factors(
        cls, evaled_factors: Iterable[EvaluatedFactor]
    ) -> Set[ScopedTerm]:
        """
        Return the set of ScopedTerm instances which span the set of
        evaluated factors.

        Args:
            evaled_factors: The evaluated factors for which to generated scoped
                terms.

        Returns:
            The scoped terms for the nominated `evaled_factors`.
        """
        scale = 1
        factors = []
        for factor in evaled_factors:
            if factor.metadata.kind is Factor.Kind.CONSTANT:
                scale *= factor.values
            elif factor.metadata.spans_intercept:
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
                    co_scoped_term.factors += (
                        ScopedFactor(factor_new.factor, reduced=False),
                    )
                    terms = cls._simplify_scoped_terms(terms)
                    combined = True
                    break
            if not combined:
                terms.append(scoped_term.copy())
        return terms

    # Methods related to looking-up, evaluating and encoding terms and factors

    def _evaluate_factor(
        self, factor: Factor, spec: ModelSpec, drop_rows: set
    ) -> EvaluatedFactor:
        if factor.expr not in self.factor_cache:
            try:
                if factor.eval_method.value == "lookup":
                    value = self._lookup(factor.expr)
                elif factor.eval_method.value == "python":
                    value = self._evaluate(factor.expr, factor.metadata, spec)
                elif factor.eval_method.value == "literal":
                    value = FactorValues(
                        self._evaluate(factor.expr, factor.metadata, spec),
                        kind=Factor.Kind.CONSTANT,
                    )
                else:
                    raise FactorEvaluationError(
                        f"The evaluation method `{factor.eval_method.value}` for factor `{factor}` is not understood."
                    )
            except FactorEvaluationError:
                raise
            except Exception as e:
                raise FactorEvaluationError(
                    f"Unable to evaluate factor `{factor}`. [{type(e).__name__}: {e}]"
                )

            if not isinstance(value, FactorValues):
                value = FactorValues(value)

            if value.__formulaic_metadata__.kind is Factor.Kind.UNKNOWN:
                if self._is_categorical(value):
                    kind = Factor.Kind.CATEGORICAL
                    spans_intercept = True
                else:
                    kind = Factor.Kind.NUMERICAL
                    spans_intercept = False

                value = FactorValues(value, kind=kind, spans_intercept=spans_intercept)

            if (
                factor.kind is not Factor.Kind.UNKNOWN
                and factor.kind is not value.__formulaic_metadata__.kind
            ):
                if factor.kind is Factor.Kind.CATEGORICAL:
                    value.__formulaic_metadata__.kind = factor.kind
                else:
                    raise FactorEncodingError(
                        f"Factor `{factor}` is expecting values of kind '{factor.kind.value}', "
                        f"but they are actually of kind '{value.__formulaic_metadata__.kind.value}'."
                    )
            if (
                factor.expr in spec.encoder_state
                and value.__formulaic_metadata__.kind
                is not spec.encoder_state[factor.expr][0]
            ):
                raise FactorEncodingError(
                    f"The model specification expects factor `{factor}` to have values of kind "
                    f"`{spec.encoder_state[factor.expr][0]}`, but they are actually of kind "
                    f"`{value.__formulaic_metadata__.kind.value}`."
                )
            self._check_for_nulls(factor.expr, value, spec.na_action, drop_rows)
            self.factor_cache[factor.expr] = EvaluatedFactor(
                factor=factor, values=value
            )
        return self.factor_cache[factor.expr]

    def _lookup(self, name):
        return self.layered_context[name]

    def _evaluate(self, expr, metadata, spec):
        return stateful_eval(
            expr, self.layered_context, {expr: metadata}, spec.transform_state, spec
        )

    def _is_categorical(self, values):
        if hasattr(values, "__formulaic_metadata__"):
            return values.__formulaic_metadata__.kind is Factor.Kind.CATEGORICAL
        return False

    def _check_for_nulls(self, name, values, na_action, drop_rows):
        pass  # pragma: no cover

    def _encode_evaled_factor(
        self,
        factor: EvaluatedFactor,
        spec: ModelSpec,
        drop_rows: set,
        reduced_rank: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(factor.values, dict) or not factor.metadata.encoded:
            if factor.expr in self.encoded_cache:
                encoded = self.encoded_cache[factor.expr]
            elif (factor.expr, reduced_rank) in self.encoded_cache:
                encoded = self.encoded_cache[(factor.expr, reduced_rank)]
            else:

                def map_dict(f):
                    """
                    This decorator allows an encoding function to operator on
                    dictionaries (which should be mapped over). This allows
                    transforms to output multiple non-encoded columns and still
                    have everything work as expected.
                    """

                    @functools.wraps(f)
                    def wrapped(values, metadata, state, *args, **kwargs):
                        if isinstance(values, dict):
                            encoded = {}
                            for k, v in values.items():
                                if isinstance(k, str) and k.startswith("__"):
                                    encoded[k] = v
                                else:
                                    nested_state = state.get(k, {})
                                    encoded[k] = wrapped(
                                        v, metadata, nested_state, *args, **kwargs
                                    )
                                    if nested_state:
                                        state[k] = nested_state
                            return encoded
                        return f(values, metadata, state, *args, **kwargs)

                    return wrapped

                # If we need to unpack values into columns, we do this here.
                # Otherwise, we pass through the original values.
                factor_values = FactorValues(
                    self._extract_columns_for_encoding(factor),
                    metadata=factor.metadata,
                )

                encoder_state = spec.encoder_state.get(factor.expr, [None, {}])[1]
                if factor.metadata.kind is Factor.Kind.CATEGORICAL:
                    encoded = map_dict(self._encode_categorical)(
                        factor_values,
                        factor.metadata,
                        encoder_state,
                        spec,
                        drop_rows,
                        reduced_rank=reduced_rank,
                    )
                elif factor.metadata.kind is Factor.Kind.NUMERICAL:
                    encoded = map_dict(self._encode_numerical)(
                        factor_values, factor.metadata, encoder_state, spec, drop_rows
                    )
                elif factor.metadata.kind is Factor.Kind.CONSTANT:
                    encoded = map_dict(self._encode_constant)(
                        factor_values, factor.metadata, encoder_state, spec, drop_rows
                    )
                else:
                    raise FactorEncodingError(
                        factor
                    )  # pragma: no cover; it is not currently possible to reach this sentinel
                spec.encoder_state[factor.expr] = (factor.metadata.kind, encoder_state)

                # Only encode once for encodings where we can just drop a field
                # later on below.
                if isinstance(encoded, dict) and factor.metadata.drop_field:
                    cache_key = factor.expr
                else:
                    cache_key = (factor.expr, reduced_rank)

                self.encoded_cache[cache_key] = encoded
        else:
            encoded = factor.values

        encoded = FactorValues(
            encoded,
            metadata=getattr(encoded, "__formulaic_metadata__", factor.metadata),
            encoded=True,
        )

        # Encoded factors will now all be dicts
        if (
            isinstance(encoded, dict)
            and encoded.__formulaic_metadata__.spans_intercept
            and reduced_rank
        ):
            encoded = FactorValues(
                encoded.copy(), metadata=encoded.__formulaic_metadata__
            )
            del encoded[encoded.__formulaic_metadata__.drop_field]

        return self._flatten_encoded_evaled_factor(factor.expr, encoded)

    def _extract_columns_for_encoding(
        self, factor: EvaluatedFactor
    ) -> Union[Any, Dict[str, Any]]:
        """
        If incoming factor has values that need to be unpacked into columns
        (e.g. a two-dimensions numpy array), do that expansion here. Otherwise,
        return the current factor values.
        """
        return as_columns(factor.values)

    def _flatten_encoded_evaled_factor(
        self, name: str, values: FactorValues[dict]
    ) -> Dict[str, Any]:
        if not isinstance(values, dict):
            return {name: values}

        # Some nested dictionaries may not be a `FactorValues[dict]` instance,
        # in which case we impute the default formatter in `FactorValues.format`.
        if hasattr(values, "__formulaic_metadata__"):
            name_format = values.__formulaic_metadata__.format
        else:
            name_format = FactorValuesMetadata.format

        flattened = {}
        for subfield, value in values.items():
            if isinstance(subfield, str) and subfield.startswith("__"):
                continue
            subname = name_format.format(name=name, field=subfield)
            if isinstance(value, dict):
                flattened.update(self._flatten_encoded_evaled_factor(subname, value))
            else:
                flattened[subname] = value

        return flattened

    @abstractmethod
    def _encode_constant(self, value, metadata, encoder_state, spec, drop_rows):
        pass  # pragma: no cover

    @abstractmethod
    def _encode_categorical(
        self, values, metadata, encoder_state, spec, drop_rows, reduced_rank=False
    ):
        pass  # pragma: no cover

    @abstractmethod
    def _encode_numerical(self, values, metadata, encoder_state, spec, drop_rows):
        pass  # pragma: no cover

    # Methods related to ModelMatrix output

    def _enforce_structure(
        self,
        cols: List[Tuple[Term, List[ScopedTerm], Dict[str, Any]]],
        spec,
        drop_rows: set,
    ) -> Generator[Tuple[Term, List[ScopedTerm], Dict[str, Any]]]:
        # TODO: Verify that imputation strategies are intuitive and make sense.
        assert len(cols) == len(spec.structure)
        for i in range(len(cols)):
            scoped_cols = cols[i][2]
            target_cols = spec.structure[i][2]
            if len(scoped_cols) > len(target_cols):
                raise FactorEncodingError(
                    f"Term `{cols[i][0]}` has generated too many columns compared to specification: generated {list(scoped_cols)}, expecting {target_cols}."
                )
            elif len(scoped_cols) < len(target_cols):
                if len(scoped_cols) == 0:
                    col = self._encode_constant(0, None, None, spec, drop_rows)
                elif len(scoped_cols) == 1:
                    col = next(iter(scoped_cols.values()))
                else:
                    raise FactorEncodingError(
                        f"Term `{cols[i][0]}` has generated insufficient columns compared to specification: generated {list(scoped_cols)}, expecting {target_cols}."
                    )
                scoped_cols = {name: col for name in target_cols}
            elif set(scoped_cols) != set(target_cols):
                raise FactorEncodingError(
                    f"Term `{cols[i][0]}` has generated columns that are inconsistent with specification: generated {list(scoped_cols)}, expecting {target_cols}."
                )

            yield cols[i][0], cols[i][1], {col: scoped_cols[col] for col in target_cols}

    def _get_columns_for_term(self, factors, spec, scale=1):
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
            out[":".join(p[0] for p in product)] = scale * functools.reduce(
                operator.mul, (p[1] for p in product)
            )
        return out

    @abstractmethod
    def _combine_columns(self, cols, spec, drop_rows):
        pass  # pragma: no cover
