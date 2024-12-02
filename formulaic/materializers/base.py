from __future__ import annotations

import ast
import functools
import inspect
import itertools
import operator
from abc import abstractmethod
from collections import defaultdict, namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from interface_meta import InterfaceMeta, inherit_docs

from formulaic.errors import (
    FactorEncodingError,
    FactorEvaluationError,
    FormulaMaterializationError,
    FormulaMaterializerInvalidError,
    FormulaMaterializerNotFoundError,
)
from formulaic.materializers.types.enums import ClusterBy, NAAction
from formulaic.materializers.types.factor_values import FactorValuesMetadata
from formulaic.model_matrix import ModelMatrices, ModelMatrix
from formulaic.parser.types import Factor, Term
from formulaic.parser.types.ordered_set import OrderedSet
from formulaic.transforms import TRANSFORMS
from formulaic.utils.cast import as_columns
from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.stateful_transforms import stateful_eval
from formulaic.utils.variables import Variable

from .types import EvaluatedFactor, FactorValues, ScopedFactor, ScopedTerm

if TYPE_CHECKING:  # pragma: no cover
    from formulaic import FormulaSpec, ModelSpec, ModelSpecs

EncodedTermStructure = namedtuple(
    "EncodedTermStructure", ("term", "scoped_terms", "columns")
)


class FormulaMaterializerMeta(InterfaceMeta):
    INTERFACE_RAISE_ON_VIOLATION = True

    REGISTERED_NAMES: Dict[str, Type[FormulaMaterializer]] = {}
    REGISTERED_INPUTS: Dict[str, List[Type[FormulaMaterializer]]] = defaultdict(list)

    def __register_implementation__(cls) -> None:
        if "REGISTER_NAME" in cls.__dict__ and cls.REGISTER_NAME:
            cls.REGISTERED_NAMES[cls.REGISTER_NAME] = cls

            if "REGISTER_INPUTS" in cls.__dict__:
                for input_type in cls.REGISTER_INPUTS:
                    cls.REGISTERED_INPUTS[input_type] = sorted(
                        cls.REGISTERED_INPUTS[input_type] + [cls],
                        key=lambda x: x.REGISTER_PRECEDENCE,
                        reverse=True,
                    )

    def for_materializer(
        cls, materializer: Union[str, FormulaMaterializer, Type[FormulaMaterializer]]
    ) -> Type[FormulaMaterializer]:
        if isinstance(materializer, str):
            if materializer not in cls.REGISTERED_NAMES:
                raise FormulaMaterializerNotFoundError(materializer)
            return cls.REGISTERED_NAMES[materializer]
        if isinstance(materializer, FormulaMaterializer):
            return type(materializer)
        if not inspect.isclass(materializer) or not issubclass(
            materializer, FormulaMaterializer
        ):
            raise FormulaMaterializerInvalidError(
                "Materializers must be subclasses of `formulaic.materializers.FormulaMaterializer`."
            )
        return materializer

    def for_data(cls, data: Any, output: Hashable = None) -> Type[FormulaMaterializer]:
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

        output_types: Set[Hashable] = set(
            *itertools.chain(
                materializer.REGISTER_OUTPUTS
                for materializer in cls.REGISTERED_INPUTS[input_type]
            )
        )
        raise FormulaMaterializerNotFoundError(
            f"No materializer has been registered for input type {repr(input_type)} that supports output type {repr(output)}. Available output types for {repr(input_type)} are: {output_types}."
        )


class FormulaMaterializer(metaclass=FormulaMaterializerMeta):
    REGISTER_NAME: Optional[str] = None
    REGISTER_INPUTS: Sequence[str] = ()
    REGISTER_OUTPUTS: Sequence[Hashable] = ()
    REGISTER_PRECEDENCE: float = 100

    # Public API

    @inherit_docs(method="_init")
    def __init__(
        self, data: Any, context: Optional[Mapping[str, Any]] = None, **params: Any
    ):
        self.data = data
        self.context = context or {}
        self.params = params
        self._init()

        self.layered_context = LayeredMapping(
            LayeredMapping(self.data_context, name="data"),
            LayeredMapping(self.context, name="context"),
            LayeredMapping(TRANSFORMS, name="transforms"),
        )

        self.factor_cache: Dict[str, EvaluatedFactor] = {}
        self.encoded_cache: Dict[Union[str, Tuple[str, bool]], Any] = {}

    def _init(self) -> None:
        pass  # pragma: no cover

    @property
    def data_context(self) -> Mapping[str, Any]:
        return self.data

    @property
    def nrows(self) -> int:
        return len(self.data)

    def get_model_matrix(
        self,
        spec: Union[FormulaSpec, ModelMatrix, ModelMatrices, ModelSpec, ModelSpecs],
        drop_rows: Optional[Set[int]] = None,
        **spec_overrides: Any,
    ) -> Union[ModelMatrix, ModelMatrices]:
        from formulaic import ModelSpec

        # Prepare ModelSpec(s)
        spec: Union[ModelSpec, ModelSpecs] = ModelSpec.from_spec(
            spec, context=self.layered_context, **spec_overrides
        )
        should_simplify = isinstance(spec, ModelSpec)
        model_specs: ModelSpecs = self._prepare_model_specs(spec)

        # Step 0: Pool all factors and transform state, ensuring consistency
        # during factor evaluation (esp. which rows get dropped).
        (
            factors,
            factor_evaluation_model_spec,
        ) = self._prepare_factor_evaluation_model_spec(model_specs)

        # Step 1: Evaluate all factors and cache the results, keeping track of
        # which rows need dropping (if `self.config.na_action == 'drop'`).
        drop_rows: Set[int] = drop_rows if drop_rows is not None else set()
        for factor in factors:
            self._evaluate_factor(factor, factor_evaluation_model_spec, drop_rows)
        drop_rows: Sequence[int] = sorted(drop_rows)

        # Step 2: Update the structured model specs with the information from
        # the shared transform state pool.
        model_specs._map(
            lambda ms: ms.transform_state.update(
                factor_evaluation_model_spec.transform_state
            )
        )

        # Step 3: Build the model matrices using the shared factor cache, and
        # by recursing over the structured model matrices.
        model_matrices = cast(
            ModelMatrices,
            model_specs._map(
                lambda model_spec: self._build_model_matrix(
                    model_spec, drop_rows=drop_rows
                ),
                as_type=ModelMatrices,
            ),
        )

        if should_simplify:
            return cast(Union[ModelMatrix, ModelMatrices], model_matrices._simplify())
        return model_matrices

    def _build_model_matrix(
        self, spec: ModelSpec, drop_rows: Sequence[int]
    ) -> ModelMatrix:
        # Step 0: Apply any requested column/term clustering
        # This must happen before Step 1 otherwise the greedy rank reduction
        # below would result in a different outcome than if the columns had
        # always been in the generated order.
        terms = self._cluster_terms(spec.formula, cluster_by=spec.cluster_by)

        # Step 1: Determine strategy to maintain structural full-rankness of output matrix
        # (reusing pre-generated structure if it is available)
        if spec.structure:
            scoped_terms_for_terms: Generator[
                Tuple[Term, Iterable[ScopedTerm]], None, None
            ] = (
                (s.term, [st.rehydrate(self.factor_cache) for st in s.scoped_terms])
                for s in spec.structure
            )
        else:
            scoped_terms_for_terms = self._get_scoped_terms(
                terms,
                ensure_full_rank=spec.ensure_full_rank,
            )

        # Step 2: Generate the columns which will be collated into the full matrix
        cols = []
        for term, scoped_terms in scoped_terms_for_terms:
            scoped_cols = {}
            for scoped_term in scoped_terms:
                if not scoped_term.factors:
                    scoped_cols["Intercept"] = (
                        scoped_term.scale
                        * self._encode_constant(1, None, {}, spec, drop_rows)
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
                                for scoped_factor in scoped_term.factors
                            ],
                            spec=spec,
                            scale=scoped_term.scale,
                        )
                    )
            cols.append((term, scoped_terms, scoped_cols))

        # Step 3: Populate remaining model spec fields
        if spec.structure:
            cols = list(self._enforce_structure(cols, spec, drop_rows))
        else:
            spec = spec.update(
                structure=[
                    EncodedTermStructure(
                        term,
                        list(st.copy(without_values=True) for st in scoped_terms),
                        list(scoped_cols),
                    )
                    for term, scoped_terms, scoped_cols in cols
                ],
            )

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

    def _prepare_model_specs(self, spec: Union[ModelSpec, ModelSpecs]) -> ModelSpecs:
        from formulaic.model_spec import ModelSpecs

        if not isinstance(spec, ModelSpecs):
            spec = ModelSpecs(spec)

        def prepare_model_spec(model_spec: ModelSpec) -> ModelSpec:
            overrides: Dict[str, Any] = {
                "materializer": self.REGISTER_NAME,
                "materializer_params": self.params,
            }

            if model_spec.output is None:
                overrides["output"] = self.REGISTER_OUTPUTS[0]
            elif model_spec.output not in self.REGISTER_OUTPUTS:
                raise FormulaMaterializationError(
                    f"Nominated output {repr(model_spec.output)} is invalid. Available output types are: {set(self.REGISTER_OUTPUTS)}."
                )

            return model_spec.update(**overrides)

        return cast(ModelSpecs, spec._map(prepare_model_spec, as_type=ModelSpecs))

    def _prepare_factor_evaluation_model_spec(
        self, model_specs: ModelSpecs
    ) -> Tuple[Set[Factor], ModelSpec]:
        from formulaic.model_spec import ModelSpec

        output = set()
        na_action = set()
        ensure_full_rank = set()
        factors: Set[Factor] = set()
        transform_state = {}

        def update_pooled_spec(model_spec: ModelSpec) -> None:
            output.add(model_spec.output)
            na_action.add(model_spec.na_action)
            ensure_full_rank.add(model_spec.ensure_full_rank)
            factors.update(
                itertools.chain(*(term.factors for term in model_spec.formula))
            )
            transform_state.update(
                model_spec.transform_state
            )  # TODO: Check for consistency?

        model_specs._map(update_pooled_spec)

        if len(output) != 1 or len(na_action) != 1 or len(ensure_full_rank) != 1:
            raise RuntimeError(
                "Provided `ModelSpec` instances are not consistent."
            )  # pragma: no cover; will only occur if users manually construct a structured model spec.

        return factors, cast(
            ModelSpec,
            ModelSpec.from_spec(
                [],
                ensure_full_rank=next(iter(ensure_full_rank)),
                na_action=next(iter(na_action)),
                output=next(iter(output)),
                transform_state=transform_state,
            ),
        )

    def _cluster_terms(
        self, terms: Sequence[Term], cluster_by: ClusterBy = ClusterBy.NONE
    ) -> Sequence[Term]:
        if cluster_by is not ClusterBy.NUMERICAL_FACTORS:
            return terms

        term_clusters = defaultdict(list)
        for term in terms:
            numerical_factors = tuple(
                factor
                for factor in term.factors
                if self.factor_cache[factor.expr].metadata.kind is Factor.Kind.NUMERICAL
            )
            term_clusters[numerical_factors].append(term)

        return [
            term for term_cluster in term_clusters.values() for term in term_cluster
        ]

    # Methods related to ensuring out matrices are structurally full-rank

    def _get_scoped_terms(
        self, terms: Iterable[Term], ensure_full_rank: bool = True
    ) -> Generator[Tuple[Term, Iterable[ScopedTerm]], None, None]:
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
        spanned: Set[ScopedTerm] = set()

        for term in terms:
            evaled_factors = [
                self.factor_cache[factor.expr]
                for factor in term.factors
                if self.factor_cache[factor.expr].values.__wrapped__ is not None
            ]
            if not evaled_factors:
                yield term, []
                continue

            if ensure_full_rank:
                term_span = (
                    self._get_scoped_terms_spanned_by_evaled_factors(evaled_factors)
                    - spanned
                )
                scoped_terms: Iterable[ScopedTerm] = self._simplify_scoped_terms(
                    term_span
                )
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
    ) -> OrderedSet[ScopedTerm]:
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
        factors: List[Tuple[Union[ScopedFactor, int], ...]] = []
        for factor in evaled_factors:
            if factor.metadata.kind is Factor.Kind.CONSTANT:
                scale *= factor.values
            elif factor.metadata.spans_intercept:
                factors.append((ScopedFactor(factor, reduced=True), 1))
            else:
                factors.append((ScopedFactor(factor),))
        return OrderedSet(
            ScopedTerm(
                factors=(cast(ScopedFactor, p) for p in prod if p != 1), scale=scale
            )
            for prod in itertools.product(*factors)
        )

    @classmethod
    def _simplify_scoped_terms(
        cls, scoped_terms: Iterable[ScopedTerm]
    ) -> OrderedSet[ScopedTerm]:
        """
        Return the minimal set of ScopedTerm instances that spans the same
        vectorspace, matching as closely as possible the intended order of the
        terms.

        This is an iterative algorithm that applies the rule:
            (anything):(reduced rank) + (anything) |-> (anything):(full rank)
        To be safe, we recurse whenever we apply the rule to make sure that
        we have fully simplified the set of terms before adding new ones.

        This is guaranteed to minimially span the vector space, keeping
        everything full-rank by avoiding overlaps.
        """
        terms: OrderedSet[ScopedTerm] = OrderedSet()
        for scoped_term in sorted(scoped_terms, key=lambda x: len(x.factors)):
            factors = set(scoped_term.factors)
            combined = False
            for existing_term in terms:
                # Check whether existing term only differs by one factor
                cofactors = set(existing_term.factors)
                factors_diff = factors - cofactors
                if len(factors) - 1 != len(cofactors) or len(factors_diff) != 1:
                    continue
                # If the different factor is a reduced factor, we can apply the
                # rule and recurse to see if there is anything else to pick up.
                factor_new = next(iter(factors_diff))
                if factor_new.reduced:
                    terms = cls._simplify_scoped_terms(
                        terms - (existing_term,)  # type: ignore
                        | (  # type: ignore
                            ScopedTerm(
                                (
                                    (
                                        ScopedFactor(factor_new.factor, reduced=False)
                                        if factor == factor_new
                                        else factor
                                    )
                                    for factor in scoped_term.factors
                                ),
                                scale=existing_term.scale * scoped_term.scale,
                            ),
                        )
                    )
                    combined = True
                    break
            if not combined:
                terms = terms | (scoped_term,)  # type: ignore
        return terms

    # Methods related to looking-up, evaluating and encoding terms and factors

    def _evaluate_factor(
        self, factor: Factor, spec: ModelSpec, drop_rows: Set[int]
    ) -> EvaluatedFactor:
        if factor.expr not in self.factor_cache:
            try:
                if factor.eval_method.value == "lookup":
                    value, variables = self._lookup(factor.expr)
                elif factor.eval_method.value == "python":
                    value, variables = self._evaluate(
                        factor.expr, factor.metadata, spec
                    )
                elif factor.eval_method.value == "literal":
                    value = FactorValues(
                        ast.literal_eval(factor.expr),
                        # self._evaluate(factor.expr, factor.metadata, spec),
                        kind=Factor.Kind.CONSTANT,
                    )
                    variables = None
                else:  # pragma: no cover; future proofing against new eval methods
                    raise FactorEvaluationError(
                        f"The evaluation method `{factor.eval_method.value}` for factor `{factor}` is not understood."
                    )
            except (
                FactorEvaluationError
            ):  # pragma: no cover; future proofing against new eval methods
                raise
            except Exception as e:
                raise FactorEvaluationError(
                    f"Unable to evaluate factor `{factor}`. [{type(e).__name__}: {e}]"
                ) from e

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
                factor=factor, values=value, variables=variables
            )
        return self.factor_cache[factor.expr]

    def _lookup(self, name: str) -> Tuple[Any, Set[Variable]]:
        sentinel = object()
        values, layer = self.layered_context.get_with_layer_name(name, default=sentinel)
        if values is sentinel:
            raise NameError(
                f"`{name}` is not present in the dataset or evaluation context."
            )
        return values, {Variable(name, roles=("value",), source=layer)}

    def _evaluate(
        self, expr: str, metadata: Any, spec: ModelSpec
    ) -> Tuple[Any, Set[Variable]]:
        variables: Set[Variable] = set()
        return (
            stateful_eval(
                expr,
                self.layered_context,
                {expr: metadata},
                spec.transform_state,
                spec,
                variables=variables,
            ),
            variables,
        )

    def _is_categorical(self, values: Any) -> bool:
        if hasattr(values, "__formulaic_metadata__"):
            return values.__formulaic_metadata__.kind is Factor.Kind.CATEGORICAL
        return False

    def _check_for_nulls(
        self, name: str, values: Any, na_action: NAAction, drop_rows: Set[int]
    ) -> None:
        pass  # pragma: no cover

    def _encode_evaled_factor(
        self,
        factor: EvaluatedFactor,
        spec: ModelSpec,
        drop_rows: Sequence[int],
        reduced_rank: bool = False,
    ) -> Dict[str, Any]:
        if not factor.metadata.encoded:
            if factor.expr in self.encoded_cache:
                encoded = self.encoded_cache[factor.expr]
            elif (factor.expr, reduced_rank) in self.encoded_cache:
                encoded = self.encoded_cache[(factor.expr, reduced_rank)]
            else:

                def map_dict(f: Any) -> Any:
                    """
                    This decorator allows an encoding function to operator on
                    dictionaries (which should be mapped over). This allows
                    transforms to output multiple non-encoded columns and still
                    have everything work as expected.
                    """

                    @functools.wraps(f)
                    def wrapped(
                        values: Any,
                        metadata: Any,
                        state: Dict[str, Any],
                        *args: Any,
                        **kwargs: Any,
                    ) -> Any:
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
                            if isinstance(values, FactorValues):
                                return FactorValues(
                                    encoded, metadata=values.__formulaic_metadata__
                                )
                            return encoded  # pragma: no cover; nothing in formulaic uses this, but is here for generality.
                        return f(values, metadata, state, *args, **kwargs)

                    return wrapped

                encoder_state: Dict[str, Any] = spec.encoder_state.get(
                    factor.expr, [None, {}]
                )[1]

                if factor.metadata.encoder is not None:
                    encoded = as_columns(
                        factor.metadata.encoder(  # type: ignore
                            factor.values,
                            reduced_rank=reduced_rank,
                            drop_rows=drop_rows,
                            encoder_state=encoder_state,
                            model_spec=spec,
                        )
                    )
                else:
                    # If we need to unpack values into columns, we do this here.
                    # Otherwise, we pass through the original values.
                    factor_values: FactorValues = FactorValues(
                        self._extract_columns_for_encoding(factor),
                        metadata=factor.metadata,
                    )

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
                            factor_values,
                            factor.metadata,
                            encoder_state,
                            spec,
                            drop_rows,
                        )
                    elif factor.metadata.kind is Factor.Kind.CONSTANT:
                        encoded = map_dict(self._encode_constant)(
                            factor_values,
                            factor.metadata,
                            encoder_state,
                            spec,
                            drop_rows,
                        )
                    else:
                        raise FactorEncodingError(
                            factor
                        )  # pragma: no cover; it is not currently possible to reach this sentinel
                spec.encoder_state[factor.expr] = (factor.metadata.kind, encoder_state)

                # Only encode once for encodings where we can just drop a field
                # later on below.
                cache_key: Union[str, Tuple[str, bool]] = (
                    factor.expr
                    if isinstance(encoded, dict) and factor.metadata.drop_field
                    else (factor.expr, reduced_rank)
                )
                self.encoded_cache[cache_key] = encoded
        else:
            encoded = as_columns(
                factor.values
            )  # pragma: no cover; we don't use this in formulaic yet.

        encoded = FactorValues(
            encoded,
            metadata=getattr(encoded, "__formulaic_metadata__", factor.metadata),
            encoded=True,
        )

        # Encoded factors will now all be dicts
        if (
            isinstance(encoded, dict)
            and encoded.__formulaic_metadata__.spans_intercept  # type: ignore
            and reduced_rank
        ):
            encoded = FactorValues(
                encoded.copy(),
                metadata=encoded.__formulaic_metadata__,  # type: ignore
                reduced=True,
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
            name_format = values.__formulaic_metadata__.get_format()
        else:
            name_format = FactorValuesMetadata.format

        flattened = {}
        for subfield, value in values.items():
            if isinstance(subfield, str) and subfield.startswith("__"):
                continue
            subname = name_format.format(name=name, field=subfield)
            if isinstance(value, dict):
                flattened.update(self._flatten_encoded_evaled_factor(subname, value))  # type: ignore
            else:
                flattened[subname] = value

        return flattened

    @abstractmethod
    def _encode_constant(
        self,
        value: Any,
        metadata: Any,
        encoder_state: Dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
    ) -> Any:
        pass  # pragma: no cover

    @abstractmethod
    def _encode_categorical(
        self,
        values: Any,
        metadata: Any,
        encoder_state: Dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
        reduced_rank: bool = False,
    ) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def _encode_numerical(
        self,
        values: Any,
        metadata: Any,
        encoder_state: Dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
    ) -> Any:
        pass  # pragma: no cover

    # Methods related to ModelMatrix output

    def _enforce_structure(
        self,
        cols: List[Tuple[Term, Iterable[ScopedTerm], Dict[str, Any]]],
        spec: ModelSpec,
        drop_rows: Sequence[int],
    ) -> Generator[Tuple[Term, Iterable[ScopedTerm], Dict[str, Any]], None, None]:
        # TODO: Verify that imputation strategies are intuitive and make sense.
        structure = cast(List[EncodedTermStructure], spec.structure)
        if not len(cols) == len(structure):  # pragma: no cover
            raise RuntimeError(
                "Specification structure and columns are mismatched. Please report this error with examples!"
            )
        for i, col_spec in enumerate(cols):
            scoped_cols = col_spec[2]
            target_cols = structure[i][2]
            if len(scoped_cols) > len(target_cols):
                raise FactorEncodingError(
                    f"Term `{col_spec[0]}` has generated too many columns compared to specification: generated {list(scoped_cols)}, expecting {target_cols}."
                )
            if len(scoped_cols) < len(target_cols):
                if len(scoped_cols) == 0:
                    col = self._encode_constant(0, None, {}, spec, drop_rows)
                elif len(scoped_cols) == 1:
                    col = tuple(scoped_cols.values())[0]
                else:
                    raise FactorEncodingError(
                        f"Term `{col_spec[0]}` has generated insufficient columns compared to specification: generated {list(scoped_cols)}, expecting {target_cols}."
                    )
                scoped_cols = {name: col for name in target_cols}
            elif set(scoped_cols) != set(target_cols):
                raise FactorEncodingError(
                    f"Term `{col_spec[0]}` has generated columns that are inconsistent with specification: generated {list(scoped_cols)}, expecting {target_cols}."
                )

            yield (
                col_spec[0],
                col_spec[1],
                {col: scoped_cols[col] for col in target_cols},
            )

    def _get_columns_for_term(
        self, factors: List[Dict[str, Any]], spec: ModelSpec, scale: float = 1
    ) -> Dict[str, Any]:
        """
        Assemble the columns for a model matrix given factors and a scale.

        This performs the row-wise Kronecker product of the factors. For greater
        compatibility with R and patsy, we reverse this product so that we
        iterate first over the latter terms.

        Args:
            factors
            scale

        Returns:
            dict
        """
        out = {}
        for reverse_product in itertools.product(
            *(factor.items() for factor in reversed(factors))
        ):
            product = reverse_product[::-1]
            out[":".join(p[0] for p in product)] = scale * functools.reduce(
                operator.mul, (p[1] for p in product)
            )
        return out

    @abstractmethod
    def _combine_columns(
        self, cols: Sequence[Tuple[str, Any]], spec: ModelSpec, drop_rows: Sequence[int]
    ) -> Any:
        pass  # pragma: no cover
