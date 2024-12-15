from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from formulaic.materializers.base import EncodedTermStructure
from formulaic.parser.types import Factor, Term
from formulaic.utils.constraints import LinearConstraints, LinearConstraintSpec
from formulaic.utils.structured import Structured
from formulaic.utils.variables import Variable

from .formula import Formula, FormulaSpec, SimpleFormula, StructuredFormula
from .materializers import ClusterBy, FormulaMaterializer, NAAction

if TYPE_CHECKING:  # pragma: no cover
    from .model_matrix import ModelMatrices, ModelMatrix
    from .transforms.contrasts import ContrastsState

# Cached property was introduced in Python 3.8 (we currently support 3.7)
try:
    from functools import cached_property
except ImportError:  # pragma: no cover
    from cached_property import cached_property  # type: ignore


@dataclass(frozen=True)
class ModelSpec:
    """
    A container for the metadata used to generate a `ModelMatrix` instance.

    This object can also be used to create a `ModelMatrix` instance that
    respects the encoding choices made during the generation of this `ModelSpec`
    instance.

    Attributes:
        Configuration:
            formula: The formula for which the model matrix was (and/or will be)
                generated.
            materializer: The materializer used (and/or to be used) to
                materialize the formula into a matrix.
            ensure_full_rank: Whether to ensure that the generated matrix is
                "structurally" full-rank (features are not included which are
                known to violate full-rankness).
            na_action: The action to be taken if NA values are found in the
                data. Can be one of: "drop" (the default), "raise" or "ignore".
            output: The desired output type (as interpreted by the materializer;
                e.g. "pandas", "sparse", etc).
            cluster_by: How to cluster terms/columns during materialization. Can
                be one of: "none" (the default) or "numerical_factors" (in which
                case terms are clustered based on their sharing of the same
                numerical factors; like patsy).

        State (these attributes are only populated during materialization):
            structure: The model matrix structure resulting from materialization.
            transform_state: The state of any stateful transformations that took
                place during factor evaluation.
            encoder_state: The state of any stateful transformations that took
                place during encoding.

    Class attributes:
        SENTINEL: Can be used as a default fallback in signatures (e.g. stateful
            transforms) to aid in typing. Must not be modified.
    """

    @classmethod
    def from_spec(
        cls,
        spec: Union[FormulaSpec, ModelMatrix, ModelMatrices, ModelSpec, ModelSpecs],
        *,
        context: Optional[Mapping[str, Any]] = None,
        **attrs: Any,
    ) -> Union[ModelSpec, ModelSpecs]:
        """
        Construct a `ModelSpec` (or `Structured[ModelSpec]`) instance for the
        nominated `spec`, setting and/or overriding any `ModelSpec` attributes
        present in `attrs`.

        Args:
            spec: The specification for which to generate a `ModelSpec`
                instance or structured set of `ModelSpec` instances.
            attrs: Any `ModelSpec` attributes to set and/or override on all
                generated `ModelSpec` instances.
            context: Optional additional context to pass through to the formula
                parsing algorithms. This is not normally required, and if
                involved operators place additional constraints on the type
                and/or structure of this context, they will raise exceptions
                when they are not satisfied with instructions for how to fix it.
        """
        from .model_matrix import ModelMatrix

        def prepare_model_spec(obj: Any) -> Union[ModelSpec, ModelSpecs]:
            if isinstance(obj, ModelMatrix):
                obj = obj.model_spec
            if isinstance(obj, ModelSpec):
                return obj.update(**attrs)
            formula = Formula.from_spec(obj, context=context)
            if isinstance(formula, StructuredFormula):
                return cast(
                    ModelSpecs, formula._map(prepare_model_spec, as_type=ModelSpecs)
                )
            return ModelSpec(formula=formula, **attrs)

        if isinstance(spec, SimpleFormula) or not isinstance(spec, Structured):
            return prepare_model_spec(spec)
        return cast(ModelSpecs, spec._map(prepare_model_spec, as_type=ModelSpecs))

    # Configuration attributes
    formula: SimpleFormula
    materializer: Optional[str] = None
    materializer_params: Optional[Dict[str, Any]] = None
    ensure_full_rank: bool = True
    na_action: NAAction = NAAction.DROP
    output: Optional[str] = None
    cluster_by: ClusterBy = ClusterBy.NONE

    # State attributes
    structure: Optional[List[EncodedTermStructure]] = None
    transform_state: Dict = field(default_factory=dict)
    encoder_state: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.__dict__["formula"] = SimpleFormula.from_spec(self.formula)

        if isinstance(self.formula, StructuredFormula):
            raise ValueError(
                "Nominated `Formula` instance has structure, which is not permitted when attaching to a `ModelSpec` instance."
            )

        # Materializer
        if self.materializer is not None and not isinstance(self.materializer, str):
            self.__dict__["materializer"] = FormulaMaterializer.for_materializer(
                self.materializer
            ).REGISTER_NAME

        # Handle string to enum mapping for values passed in during instantiation
        self.__dict__["na_action"] = NAAction(self.na_action)
        self.__dict__["cluster_by"] = ClusterBy(self.cluster_by)

    # Derived features

    @property
    def __structure(self) -> List[EncodedTermStructure]:
        """
        A reference to `.structure` if it is populated, or otherwise an
        exception is raised.
        """
        if self.structure is None:
            raise RuntimeError(
                "`ModelSpec.structure` has not yet been populated. This will "
                "likely be resolved by using the `ModelSpec` instance attached "
                "to the model matrix generated when calling `.get_model_matrix()`."
            )
        return self.structure

    @cached_property
    def column_names(self) -> Sequence[str]:
        """
        The names associated with the columns of the generated model matrix.
        """
        return tuple(feature for row in self.__structure for feature in row.columns)

    @cached_property
    def column_indices(self) -> Dict[str, int]:
        """
        An ordered mapping from column names to the column index in generated
        model matrices.
        """
        return {name: i for i, name in enumerate(self.column_names)}

    def get_column_indices(self, columns: Union[str, Sequence[str]]) -> List[int]:
        """
        Generate a list of column indices corresponding to the nominated column
        names. This is useful when you want to slice a model matrix by specific
        columns, and do not want to have to generate the indices yourself.

        Args:
            columns: The column names to include in the subset.
        """
        if isinstance(columns, str):
            columns = [columns]
        return [self.column_indices[column] for column in columns]

    @property
    def terms(self) -> List[Term]:
        """
        The terms used to generate model matrices from this `ModelSpec`
        instance.
        """
        return list(self.formula)

    @cached_property
    def term_indices(self) -> Dict[Term, List[int]]:
        """
        An ordered mapping of `Term` instances to the generated column indices.

        Note: Since terms hash using their string representation, you can look
        up elements of this mapping using the string representation of the
        `Term`.
        """
        slices = {}
        start = 0
        for row in self.__structure:
            end = start + len(row[2])
            slices[row[0]] = list(range(start, end))
            start = end
        return slices

    def get_term_indices(
        self, terms_spec: FormulaSpec, **formula_kwargs: Any
    ) -> List[int]:
        """
        Generate a list of column indices corresponding to the columns
        associated with the nominated `term_spec`.

        This is useful when you want to slice a model matrix by specific terms.
        If you want to generate new matrices for term subsets, consider using
        `ModelSpec.subset()` instead.

        The nominated `terms_spec` will be interpreted as a formula
        specification, and the resulting term set must only include terms that
        are present in this `ModelSpec` instance. A `ValueError` error will be
        raised if the `terms_spec` is structured or contains terms not
        represented by this `ModelSpec`.

        The indices will be ordered according to the order of the terms in the
        `terms_spec`.

        Args:
            terms_spec: The specification for the terms for which to extract
                indices.
            formula_kwargs: Additional keyword arguments to pass to the
                `Formula.from_spec` constructor to control (e.g.) ordering.
        """
        terms: List[Term] = list(
            self.__get_restricted_formula(terms_spec, **formula_kwargs)
        )
        return [idx for term in terms for idx in self.term_indices[term]]

    @cached_property
    def term_slices(self) -> Dict[Term, slice]:
        """
        An ordered mapping of `Term` instances to a slice that when used on
        the columns of the model matrix will subsample the model matrix down to
        those corresponding to each term.

        Note: Since terms hash using their string representation, you can look
        up elements of this mapping using the string representation of the
        `Term`.
        """
        return {
            k: slice(v[0], v[-1] + 1) if v else slice(0, 0)
            for k, v in self.term_indices.items()
        }

    @cached_property
    def term_factors(self) -> Dict[Term, Set[Factor]]:
        """
        A mapping from `Term` instances to the factors which were used to
        generate them.
        """
        term_factors: Dict[Term, Set[Factor]] = defaultdict(set)
        for term in self.terms:
            for factor in term.factors:
                term_factors[term].add(factor)
        return dict(term_factors)

    @cached_property
    def term_variables(self) -> Dict[Term, Set[Variable]]:
        """
        An ordered mapping of `Term` instances to the set of `Variable`
        instances corresponding to the variables used in the evaluation of that
        term. `Variable` instances are enriched strings, with the additional
        attributes `.roles` and `.source`.
        """
        term_variables = {}
        for row in self.__structure:
            term_variables[row[0]] = Variable.union(
                *(term.variables for term in row[1]),
            )
        return term_variables

    @cached_property
    def factors(self) -> Set[Factor]:
        """
        The factors used to generate model matrices from this `ModelSpec`
        instance.
        """
        return {factor for term in self.terms for factor in term.factors}

    @cached_property
    def factor_terms(self) -> Dict[Factor, Set[Term]]:
        """
        A mapping from `Factor` instances to the terms which used it. This is
        the reverse mapping of `.term_factors`.
        """
        factor_terms: Dict[Factor, Set[Term]] = defaultdict(set)
        for term, factors in self.term_factors.items():
            for factor in factors:
                factor_terms[factor].add(term)
        return dict(factor_terms)

    @cached_property
    def factor_variables(self) -> Dict[Factor, Set[Variable]]:
        """
        A mapping from `Factor` instances to the variables used in the evaluation
        of that factor.
        """
        factor_variables: Dict[Factor, List[Variable]] = defaultdict(list)
        for s in self.__structure:
            for scoped_term in s.scoped_terms:
                for scoped_factor in scoped_term.factors:
                    factor_variables[scoped_factor.factor.factor].extend(
                        scoped_factor.factor.variables
                    )

        return {
            factor: Variable.union(factor_variables.get(factor, []))
            for factor in self.factors
        }

    @cached_property
    def factor_contrasts(self) -> Dict[Factor, ContrastsState]:
        """
        A mapping of `Factor` instances to their contrasts state. This is useful
        if you would like to introspect some of the coding choices, or reuse
        these encodings outside of formulaic. Only categorical factors that were
        encoded by Formulaic will be included in this mapping.

        Note that these contrast states do *not* include whether the factor was
        encoded using reduced rank or not, since this is potentially ambiguous
        even within a single term. [Depending on the context, a factor may
        be encoded as either or both full and reduced rank in order to fully
        span the vector space]. Instead, you can choose whether to reduce the
        rank using:
        ```
        model_spec.factor_contrasts[<factor>].get_coding_matrix(reduced_rank=True)
        ```
        If not specified, the default is to reduce the rank, which gives the
        more interesting matrices.

        Refer to the documentation of `ContrastsState` for more details.
        """
        return {
            factor: self.encoder_state[factor][1]["contrasts"]
            for factor in self.factors
            if factor in self.encoder_state
            and self.encoder_state[factor][0] is Factor.Kind.CATEGORICAL
            and "contrasts" in self.encoder_state[factor][1]
        }

    @cached_property
    def variables(self) -> Set[Variable]:
        """
        The variables used during the materialization of the entire formula.
        """
        return Variable.union(
            *(variables for variables in self.term_variables.values())
        )

    @cached_property
    def variable_terms(self) -> Dict[Variable, Set[Term]]:
        """
        A mapping from `Variable` instances to the terms which used it. This is
        the reverse mapping of `.term_variables`.
        """
        variable_terms: Dict[Variable, Set[Term]] = defaultdict(set)
        for term, variables in self.term_variables.items():
            for variable in variables:
                variable_terms[variable].add(term)
        return dict(variable_terms)

    @cached_property
    def variable_indices(self) -> Dict[Variable, List[int]]:
        """
        A mapping from `Variable` instances to the indices in the model matrix
        where they were used.
        """
        return {
            variable: sorted(
                {index for term in terms for index in self.term_indices[term]}
            )
            for variable, terms in self.variable_terms.items()
        }

    def get_variable_indices(
        self, variables: Sequence[Union[str, Variable]]
    ) -> List[int]:
        """
        Generate a list of column indices corresponding to the columns associated
        with the nominated variables. This is useful when you want to slice a model
        matrix by specific variables, and do not want to have to generate the indices
        yourself.

        Args:
            variables: The variable names to include in the subset.
        """
        return [
            idx
            for variable in variables
            for idx in self.variable_indices[variable]  # type: ignore # Variables are strings too
        ]

    @cached_property
    def variables_by_source(self) -> Dict[Optional[str], Set[Variable]]:
        """
        A mapping of source name to the set of variables drawn from that source.
        Formulaic, by default, has three top-level sources of variables:
        'data', 'transforms', and 'context'.
        """
        variables_by_source: Dict[Optional[str], Set[Variable]] = defaultdict(set)
        for variable in self.variables:
            variables_by_source[variable.source].add(variable)
        return dict(variables_by_source)

    @property
    def required_variables(self) -> Set[Variable]:
        """
        The set of variables required to be in the data to materialize this
        model specification.

        If `.structure` has not been populated (which contains metadata about
        which columns where ultimate drawn from the data during
        materialization), then this will fallback to the variables inferred to
        be required by `.formula`.
        """
        if self.structure is None:
            return self.formula.required_variables
        return self.variables_by_source.get("data", set())

    def get_slice(self, columns_identifier: Union[int, str, Term, slice]) -> slice:
        """
        Generate a `slice` instance corresponding to the columns associated with
        the nominated `columns_identifier`. While this is provided for
        convenience, it is usually better in library code to directly use the
        indexing metadata methods/attributes associated with the nominated
        identifier.

        Args:
            columns_identifier: The identifier for which the slice should be
                generated. Can be one of:
                    - an integer specifying a specific column index.
                    - a `Term` instance
                    - a string representation of a term
                    - a column name
        """
        if isinstance(columns_identifier, slice):
            return columns_identifier
        if isinstance(columns_identifier, int):
            return slice(columns_identifier, columns_identifier + 1)

        term_slices = self.term_slices
        if isinstance(columns_identifier, Term):
            if columns_identifier not in term_slices:
                raise ValueError(
                    f"Model matrices built using this spec do not include term: `{columns_identifier}`."
                )
            return term_slices[columns_identifier]
        if columns_identifier in term_slices:
            return term_slices[columns_identifier]  # type: ignore # Terms hash equivalent to their string repr

        column_indices = self.column_indices
        if columns_identifier in column_indices:
            idx = column_indices[columns_identifier]
            return slice(idx, idx + 1)

        raise ValueError(
            f"Model matrices built using this spec do not have any columns related to: `{repr(columns_identifier)}`."
        )

    # Utility methods

    def get_materializer(
        self, data: Any, context: Optional[Mapping[str, Any]] = None
    ) -> FormulaMaterializer:
        """
        Construct a `FormulaMaterializer` instance for `data` that can be used
        to generate model matrices consistent with this model specification.

        Args:
            data: The data for which to build the materializer.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
        """
        if self.materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(self.materializer)
        return materializer(data, context=context, **(self.materializer_params or {}))

    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        drop_rows: Optional[Set[int]] = None,
        **attr_overrides: Any,
    ) -> ModelMatrix:
        """
        Build the model matrix (or matrices) realisation of this model spec for
        the nominated `data`.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            drop_rows: An optional set of row indices to drop from the model
                matrix. If specified, it will also be updated during
                materialization with any additional rows dropped due to null
                values.
            attr_overrides: Any `ModelSpec` attributes to override before
                constructing model matrices. This is shorthand for first
                running `ModelSpec.update(**attr_overrides)`.
        """
        if attr_overrides:
            return self.update(**attr_overrides).get_model_matrix(data, context=context)
        return cast(
            "ModelMatrix",
            self.get_materializer(data, context=context).get_model_matrix(
                self, drop_rows=drop_rows
            ),
        )

    def get_linear_constraints(self, spec: LinearConstraintSpec) -> LinearConstraints:
        """
        Construct a `LinearConstraints` instance from a specification based on
        the structure of the model matrices associated with this model spec.

        Args:
            spec: The specification from which to derive the constraints. Refer
                to `LinearConstraints.from_spec` for more details.
        """
        return LinearConstraints.from_spec(spec, variable_names=self.column_names)

    # Transforms

    def update(self, **kwargs: Any) -> ModelSpec:
        """
        Create a copy of this `ModelSpec` instance with the nominated attributes
        mutated.
        """
        return replace(self, **kwargs)

    def subset(self, terms_spec: FormulaSpec, **formula_kwargs: Any) -> ModelSpec:
        """
        Subset this `ModelSpec` instance to only include the columns associated
        with the nominated terms.

        This is useful when you want to fit restricted models on a strict subset
        of features included in this `ModelSpec` instance, and want to generate
        new model matrices with just these terms.  If you just want to subset an
        existing model matrix, you can use `ModelSpec.get_term_indices()`
        instead.

        Terms are selected from this `ModelSpec` instance by constructing a
        `Formula` instance from the provided `terms_spec`, and then matching the
        terms with those found in this `ModelSpec` instance. An error will be
        raised if the `terms_spec` is incompatibly structured or contains terms
        not represented by this `ModelSpec` instance. The model spec column
        ordering will follow the ordering of the terms in `terms_spec`.

        Note that subsetting this `ModelSpec` is in general not equivalent to
        constructing this `ModelSpec` instance from scratch with the provided
        formula specification. Specifically, the output is likely not to be
        structurally full-rank whenever categorical variables are involved.
        Instead, columns generated from the subset model spec are guaranteed to
        match the corresponding columns generated from this parent model spec.

        Args:
            terms_spec: The terms to include in the subset. A `Formula` instance
                will be constructed from this specification, and the resulting
                terms will be used to select the terms to include from this
                model spec.
            formula_kwargs: Additional keyword arguments to pass to the
                `Formula.from_spec` constructor to control (e.g.) ordering.
        """

        formula: SimpleFormula = self.__get_restricted_formula(
            terms_spec, **formula_kwargs
        )
        terms: List[Term] = list(formula)
        terms_set: Set[Term] = set(terms)
        term_structure = {s.term: s for s in self.__structure if s.term in terms_set}

        return self.update(
            formula=formula,
            structure=[term_structure[term] for term in terms],
        )

    def differentiate(self, *wrt: str, use_sympy: bool = False) -> ModelSpec:
        """
        EXPERIMENTAL: Take the gradient of this model spec. When used a linear
        regression, evaluating a trained model on model matrices generated by
        this formula is equivalent to estimating the gradient of that fitted
        form with respect to `vars`.

        Args:
            wrt: The variables with respect to which the gradient should be
                taken.
            use_sympy: Whether to use sympy to perform symbolic differentiation.

        Notes:
            This method is provisional and may be removed in any future major
            version.
        """
        return self.update(
            formula=self.formula.differentiate(*wrt, use_sympy=use_sympy),
        )

    # Only include dataclass fields when pickling.
    def __getstate__(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() if k in self.__dataclass_fields__
        }

    # Helpers

    def __get_restricted_formula(
        self, spec: FormulaSpec, **formula_kwargs: Any
    ) -> SimpleFormula:
        """
        Construct a `Formula` instance from the provided `spec` that is a
        restriction of the formula associated with this `ModelSpec` instance. A
        `ValueError` is raised if the provided `spec` results in a structured
        formula, or if it contains terms that are not described by this
        `ModelSpec`.

        Args:
            spec: The formula specification for the restricted formula.
            formula_kwargs: Additional keyword arguments to pass to the
                `Formula.from_spec` constructor to control (e.g.) ordering.
        """
        formula = SimpleFormula.from_spec(spec, **formula_kwargs)
        if isinstance(formula, StructuredFormula):
            raise ValueError(
                "Cannot subset a `ModelSpec` using a formula that has structure."
            )

        missing_terms: Set[Term] = set(formula).difference(self.terms)
        if missing_terms:
            raise ValueError(
                f"Cannot subset a model spec with terms not present in the original model spec: {missing_terms}."
            )

        return formula


class ModelSpecs(Structured[ModelSpec]):
    """
    A `Structured[ModelSpec]` subclass that exposes some convenience methods
    that should be mapped onto all contained `ModelSpec` instances.
    """

    def _prepare_item(self, key: str, item: Any) -> Any:
        # Verify that all included items are `ModelSpec` instances.
        if not isinstance(item, ModelSpec):
            raise TypeError(
                "`ModelSpecs` instances expect all items to be instances of "
                f"`ModelSpec`. [Got: {repr(item)} of type {repr(type(item))} "
                f"for key {repr(key)}."
            )
        return item

    @property
    def required_variables(self) -> Set[Variable]:
        """
        The set of variables required to be in the data to materialize all of
        the model specifications in this `ModelSpecs` instance.
        """
        variables: Set[Variable] = set()
        self._map(lambda ms: variables.update(ms.required_variables))
        return variables

    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        drop_rows: Optional[Set[int]] = None,
        **attr_overrides: Any,
    ) -> ModelMatrices:
        """
        This method proxies the `ModelSpec.get_model_matrix(...)` API and allows
        it to be called on a structured set of `ModelSpec` instances. If all
        `ModelSpec.materializer` and `ModelSpec.materializer_params` values are
        unset or the same, then they are jointly evaluated allowing re-use of
        the same cached across the specs.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            drop_rows: An optional set of row indices to drop from the model
                matrix. If specified, it will also be updated during
                materialization with any additional rows dropped due to null
                values.
            attr_overrides: Any `ModelSpec` attributes to override before
                constructing model matrices. This is shorthand for first
                running `ModelSpec.from_spec(model_specs, **attr_overrides)`.
        """
        from formulaic import ModelMatrices

        if attr_overrides:
            return ModelSpec.from_spec(self, **attr_overrides).get_model_matrix(
                data, context=context, drop_rows=drop_rows
            )

        # Check whether we can generate model matrices jointly (i.e. all
        # materializers and their params are the same)
        jointly_generate = False
        materializer, materializer_params = None, None

        for spec in self._flatten():
            if not spec.materializer:
                continue
            if materializer not in (
                None,
                spec.materializer,
            ) or materializer_params not in (
                None,
                spec.materializer_params,
            ):
                break
            materializer, materializer_params = (
                spec.materializer,
                spec.materializer_params or None,
            )
        else:
            jointly_generate = True

        if jointly_generate:
            if materializer is None:
                materializer = FormulaMaterializer.for_data(data)
            else:
                materializer = FormulaMaterializer.for_materializer(materializer)
            return materializer(  # type: ignore
                data, context=context, **(materializer_params or {})
            ).get_model_matrix(self)

        return cast(
            ModelMatrices,
            self._map(
                lambda model_spec: model_spec.get_model_matrix(
                    data, context=context, drop_rows=drop_rows
                ),
                as_type=ModelMatrices,
            ),
        )

    def subset(self, terms_spec: FormulaSpec) -> ModelSpecs:
        """
        Subset this `ModelSpecs` instance to only include the columns associated
        with the nominated terms. The structure of `terms_spec` must match the
        structure of this `ModelSpecs` instance where they overlap. For more
        details, please reference to `ModelSpec.subset`.

        Args:
            terms_spec: The terms to include in the subset. A `Formula` instance
                will be constructed from this specification, and the resulting
                terms will be used to select the terms to include from this
                model spec.
        """

        def map_formula_structure_onto_model_spec(
            formula: SimpleFormula, context: Tuple[Union[int, str], ...]
        ) -> ModelSpec:
            try:
                return self[context].subset(formula)
            except KeyError:
                raise ValueError(
                    f"Cannot subset a `ModelSpecs` instance using a formula with a different structure [indexing path `{context}` not found]."
                )

        formula = SimpleFormula.from_spec(terms_spec)
        if isinstance(formula, SimpleFormula):
            raise ValueError(
                "Formula has no structure, and hence does not match the structure of the `ModelSpec` instance."
            )
        return cast(
            ModelSpecs,
            formula._map(map_formula_structure_onto_model_spec, as_type=ModelSpecs),
        )

    def differentiate(self, *wrt: str, use_sympy: Any = False) -> ModelSpecs:
        """
        This method proxies the experimental `ModelSpec.differentiate(...)` API.
        See `ModelSpec.differentiate` for more details.
        """
        return cast(
            ModelSpecs,
            self._map(
                lambda model_spec: model_spec.differentiate(*wrt, use_sympy=use_sympy),
                as_type=ModelSpecs,
            ),
        )
