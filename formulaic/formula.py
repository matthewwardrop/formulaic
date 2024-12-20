from __future__ import annotations

import sys
from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import Self, TypeAlias

from formulaic.utils.sentinels import MISSING, MissingType

from .errors import FormulaInvalidError
from .model_matrix import ModelMatrix
from .parser import DefaultFormulaParser
from .parser.types import FormulaParser, OrderedSet, Term
from .utils.calculus import differentiate_term
from .utils.deprecations import deprecated
from .utils.structured import Structured
from .utils.variables import Variable, get_expression_variables

FormulaSpec: TypeAlias = Union[
    "Formula",
    str,
    List[Union[str, Term]],
    Set[Union[str, Term]],
    Dict[str, "FormulaSpec"],
    Tuple["FormulaSpec", ...],
    Structured["FormulaSpec"],
]
_SelfType = TypeVar("_SelfType", bound="Formula")


DEFAULT_PARSER = DefaultFormulaParser()
DEFAULT_NESTED_PARSER = DefaultFormulaParser(include_intercept=False)


class OrderingMethod(Enum):
    NONE = "none"
    DEGREE = "degree"
    SORT = "sort"


class _FormulaMeta(ABCMeta):
    """
    This metaclass serves two purposes:
    (1) to allow the `Formula` class constructor to delegate in the construction
        of either `SimpleFormula` or `StructuredFormula` subclass instances
        based on the input specifications; without the `__init__` constuctor
        being called twice.
    (2) to provide the generic `.from_spec()` constructor that is only
        accessible on the base classes, and not instances, of `Formula`.
    """

    def __call__(
        cls,
        root: Union[FormulaSpec, MissingType] = MISSING,
        *,
        _ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
        _parser: Optional[FormulaParser] = None,
        _nested_parser: Optional[FormulaParser] = None,
        _context: Optional[Mapping[str, Any]] = None,
        **structure: FormulaSpec,
    ) -> Formula:
        """
        Construct a `Formula` subclass instance based on the provided
        specifications. If the resulting formula has no structure, a
        `SimpleFormula` instance will be returned; otherwise, a
        `StructuredFormula`.

        Some arguments are prefixed with underscores to prevent collision with
        formula structure.

        Args:
            root: The (root) formula specification.
            _parser: The `FormulaParser` instance to use when parsing complete
                formulae (vs. individual terms). If not specified,
                `DefaultFormulaParser()` is used.
            _nested_parser: The `FormulaParser` instance to use when parsing
                strings describing nested or individual terms (e.g. when `spec`
                is a list of string term identifiers). If not specified and
                `parser` is specified, `parser` is used; if `parser` is not
                specified, `DefaultFormulaParser(include_intercept=False)` is
                used instead.
            _ordering: The ordering method to apply to the terms implied by the
                formula `spec`. Can be: "none", "degree" (default), or "sort".
            structure: Additional structure to be passed to the
                `StructuredFormula` constructor.
        """
        if cls is not Formula:
            self: Formula = cls.__new__(cls)  # type: ignore
            self.__init__(  # type: ignore
                root,
                _ordering=_ordering,
                _parser=_parser,
                _nested_parser=_nested_parser,
                _context=_context,
                **structure,
            )
            return self

        if root is MISSING and not structure:
            return SimpleFormula([])
        if structure:
            return StructuredFormula(
                root,
                _parser=_parser,
                _nested_parser=_nested_parser,
                _ordering=_ordering,
                _context=_context,
                **structure,  # type: ignore[arg-type]
            )._simplify()
        return cls.from_spec(
            cast(FormulaSpec, root),
            ordering=_ordering,
            parser=_parser,
            nested_parser=_nested_parser,
            context=_context,
        )

    def from_spec(
        cls,
        spec: FormulaSpec,
        *,
        ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
        parser: Optional[FormulaParser] = None,
        nested_parser: Optional[FormulaParser] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Union[SimpleFormula, StructuredFormula]:
        """
        Construct a `SimpleFormula` or `StructuredFormula` instance from a
        specification.

        Args:
            spec: The formula specification.
            parser: The `FormulaParser` instance to use when parsing complete
                formulae (vs. individual terms). If not specified,
                `DefaultFormulaParser()` is used.
            nested_parser: The `FormulaParser` instance to use when parsing
                strings describing nested or individual terms (e.g. when `spec`
                is a list of string term identifiers). If not specified and
                `parser` is specified, `parser` is used; if `parser` is not
                specified, `DefaultFormulaParser(include_intercept=False)` is
                used instead.
            ordering: The ordering method to apply to the terms implied by the
                formula `spec`. Can be: "none", "degree" (default), or "sort".
        """
        if isinstance(spec, Formula):
            return cast(Union[SimpleFormula, StructuredFormula], spec)

        nested_parser = nested_parser or parser or DEFAULT_NESTED_PARSER
        parser = parser or DEFAULT_PARSER

        if isinstance(spec, str):
            spec = cast(
                FormulaSpec,
                (parser or DefaultFormulaParser())
                .get_terms(spec, context=context)
                ._simplify(),
            )

        if isinstance(spec, dict):
            return StructuredFormula(
                _parser=parser,
                _nested_parser=nested_parser,
                _ordering=ordering,
                _context=context,
                **spec,  # type: ignore[arg-type]
            )
        if isinstance(spec, Structured):
            return StructuredFormula(
                _ordering=ordering,
                _parser=nested_parser,
                _nested_parser=nested_parser,
                _context=context,
                **spec._structure,
            )._simplify()
        if isinstance(spec, tuple):
            return StructuredFormula(
                spec,
                _ordering=ordering,
                _parser=parser,
                _nested_parser=nested_parser,
                _context=context,
            )._simplify()
        if isinstance(spec, (list, set, OrderedSet)):
            terms = [
                term
                for value in spec
                for term in (
                    nested_parser.get_terms(value, context=context)  # type: ignore[attr-defined]
                    if isinstance(value, str)
                    else [value]
                )
            ]
            return SimpleFormula(terms, _ordering=ordering)
        raise FormulaInvalidError(f"Unrecognized formula specification: {repr(spec)}.")


class Formula(metaclass=_FormulaMeta):
    """
    The base class for all formulae represented by Formulaic. This class can be
    directly instantiated, which will result in the construction of an
    appropriate subclass instance (either `SimpleFormula` or
    `StructuredFormula`), depending on whether the resulting formula has
    structure or not.

    The atomic element of all formulae is a `SimpleFormula`, which in turn is a
    mutable sequence of arbitrarily many `Term` instances. `StructuredFormula`
    allows the composition of multiple `SimpleFormula` instances into a
    structured result.

    Examples:
    ```
    >>> Formula("x")  # -> SimpleFormula
    1 + x
    >>> Formula("y ~ x")  # -> StructuredFormula
    .lhs:
        y
    .rhs:
        1 + x
    >>> Formula("x + y", a=["x", "y:z"], b="y ~ z")  # -> StructuredFormula
    root:
        1 + x + y
    .a:
        x + y:z
    .b:
        .lhs:
            y
        .rhs:
            z
    ```

    You can control how strings are parsed into formulae by passing in custom
    parsers via `_parser` and `_nested_parser`. For example:
    ```
    >>> Formula("y ~ x", _parser=DefaultFormulaParser(include_intercept=False))
    .lhs:
        y
    .rhs:
        x
    ```
    """

    @abstractmethod
    def __init__(
        self,
        root: Union[FormulaSpec, MissingType] = MISSING,
        *,
        _parser: Optional[FormulaParser] = None,
        _nested_parser: Optional[FormulaParser] = None,
        _ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
        _context: Optional[Mapping[str, Any]] = None,
        **structure: FormulaSpec,
    ):
        """
        This constructor is never actually called, but documents the arguments
        that should be used by all subclass constructors. These will be
        dispatched to the appropriate subclass constructor via the metaclass
        and/or `Formula.from_spec`.
        """

    @abstractmethod
    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        drop_rows: Optional[Set[int]] = None,
        **spec_overrides: Any,
    ) -> Union[ModelMatrix, Structured[ModelMatrix]]:
        """
        Build the model matrix (or matrices) realisation of this formula for the
        nominated `data`.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            drop_rows: An optional set of row indices to drop from the model
                matrix. If specified, it will also be updated during
                materialization with any additional rows dropped due to null
                values.
            spec_overrides: Any `ModelSpec` attributes to set/override. See
                `ModelSpec` for more details.
        """

    @property
    @abstractmethod
    def required_variables(self) -> Set[Variable]:
        """
        The set of variables required to be in the data to materialize this
        formula.

        Attempts are made to restrict these variables only to those expected in
        the data, and not, for example, those associated with transforms and/or
        values present in the evaluation namespace by default (e.g. `y ~ C(x)`
        would include only `y` and `x`). This may not always be possible for
        more advanced formulae that insert constants into the formula via the
        evaluation context rather than the data context.
        """

    @abstractmethod
    def differentiate(
        self: _SelfType,
        *wrt: str,
        use_sympy: bool = False,
    ) -> _SelfType:
        """
        Take the gradient of this formula with respect to the variables in
        `wrt`.

        When used a linear regression context, making predictions based on the
        model matrices generated the differentiated formula is equivalent to
        estimating the gradient of the fitted model with respect to `wrt`.

        Args:
            wrt: The variables with respect to which the gradient should be
                taken.
            use_sympy: Whether to use sympy to perform symbolic differentiation.
        """


class SimpleFormula(
    MutableSequence[Term] if sys.version_info >= (3, 9) else MutableSequence,  # type: ignore
    Formula,
):
    """
    The atomic component of all formulae represented by Formulaic, which in turn
    is a mutable sequence of `Term` instances. `StructuredFormula` uses
    `SimpleFormula` as its nodes.

    Instances of this class can be used directly as a mutable sequence of
    `Term`s, including indexing and iteration. Mutations to the sequence will
    trigger reordering of the terms according to the specified ordering method.

    Attributes:
        ordering: The ordering method to use for the terms in this container (
            passed as `_ordering` to the `SimpleFormula` constructor).

    Note: This class' constructor is not intended to be called directly by users
    in standard usage, and requires that the specification passed by users is
    already a iterable sequence of `Term` instances. This class is not capable
    of parsing formulae from strings, and will raise an error if a string is
    passed.
    """

    def __init__(
        self,
        root: Union[Iterable[Term], MissingType] = MISSING,
        *,
        _ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
        _parser: Optional[FormulaParser] = None,
        _nested_parser: Optional[FormulaParser] = None,
        _context: Optional[Mapping[str, Any]] = None,
        **structure: FormulaSpec,
    ):
        if root is MISSING:
            root = ()
        if isinstance(root, str) or not isinstance(root, Iterable):
            raise FormulaInvalidError(
                "`SimpleFormula` should be constructed from a list of `Term` instances, "
                "not a string. To parse a formula string or other specifications, "
                "please use `Formula` or `StructuredFormula` instead."
            )
        if structure:
            raise FormulaInvalidError(
                "`SimpleFormula` does not support nested structure. To create a "
                "structured formula, use `StructuredFormula` instead."
            )
        self.__terms = list(root)
        self.ordering = OrderingMethod(_ordering)

        self.__validate_terms(self.__terms)

        self._reorder()

    @classmethod
    def __validate_terms(cls, terms: Any) -> None:
        """
        Verify that `terms` is a valid sequence of `Term` instances.
        """
        if not all(isinstance(t, Term) for t in terms):
            raise FormulaInvalidError(
                f"All components of a `SimpleFormula` should be `Term` instances. Found: {repr(terms)}. To use formula strings, please use `Formula` or `StructuredFormula` instead."
            )

    def _reorder(self, ordering: Optional[OrderingMethod] = None) -> None:
        """
        Reorder the terms in this container in-place according to the specified
        ordering method.

        Args:
            ordering: The ordering method to use for the terms in this container.
                If not specified, the default ordering method for this container
                is used.
        """
        ordering = OrderingMethod(ordering if ordering is not None else self.ordering)
        orderer = None
        if ordering is OrderingMethod.DEGREE:
            orderer = lambda terms: sorted(terms, key=lambda term: term.degree)
        elif ordering is OrderingMethod.SORT:
            orderer = lambda terms: sorted(
                [Term(factors=sorted(term.factors)) for term in terms]
            )

        if orderer is not None:
            self.__terms = orderer(self.__terms)

    # MutableSequence implementation

    @overload
    def __getitem__(self, key: int) -> Term: ...

    @overload
    def __getitem__(self, key: slice) -> SimpleFormula: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[Term, SimpleFormula]:
        if isinstance(key, slice):
            return self.__class__(self.__terms[key], _ordering=self.ordering)
        else:
            return self.__terms[key]

    @overload
    def __setitem__(self, key: int, value: Term) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[Term]) -> None: ...

    def __setitem__(self, key, value):  # type: ignore
        self.__validate_terms([value])
        self.__terms[key] = value
        self._reorder()

    @overload
    def __delitem__(self, key: int) -> None: ...

    @overload
    def __delitem__(self, key: slice) -> None: ...

    def __delitem__(self, key):  # type: ignore
        del self.__terms[key]

    def __len__(self) -> int:
        return len(self.__terms)

    def insert(self, index: int, value: Term) -> None:
        self.__validate_terms([value])
        self.__terms.insert(index, value)
        self._reorder()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SimpleFormula):
            other = list(other)
        if isinstance(other, list):
            return self.__terms == other
        return NotImplemented

    # Transforms

    def differentiate(  # pylint: disable=redefined-builtin
        self,
        *wrt: str,
        use_sympy: bool = False,
    ) -> SimpleFormula:
        """
        Take the gradient of this formula with respect to the variables in
        `wrt`.

        When used a linear regression context, making predictions based on the
        model matrices generated the differentiated formula is equivalent to
        estimating the gradient of the fitted model with respect to `wrt`.

        Args:
            wrt: The variables with respect to which the gradient should be
                taken.
            use_sympy: Whether to use sympy to perform symbolic differentiation.
        """
        return SimpleFormula(
            [
                differentiate_term(term, wrt, use_sympy=use_sympy)
                for term in self.__terms
            ],
            # Preserve term ordering even if differentiation modifies degrees/etc.
            _ordering=OrderingMethod.NONE,
        )

    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        drop_rows: Optional[Set[int]] = None,
        **spec_overrides: Any,
    ) -> Union[ModelMatrix, Structured[ModelMatrix]]:
        """
        Build the model matrix (or matrices) realisation of this formula for the
        nominated `data`.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            drop_rows: An optional set of row indices to drop from the model
                matrix. If specified, it will also be updated during
                materialization with any additional rows dropped due to null
                values.
            spec_overrides: Any `ModelSpec` attributes to set/override. See
                `ModelSpec` for more details.
        """
        from .model_spec import ModelSpec

        return ModelSpec.from_spec(self, **spec_overrides).get_model_matrix(
            data, context=context, drop_rows=drop_rows
        )

    @property
    def required_variables(self) -> Set[Variable]:
        """
        The set of variables required in the data order to materialize this
        formula.

        Attempts are made to restrict these variables only to those expected in
        the data, and not, for example, those associated with transforms and/or
        values present in the evaluation namespace by default (e.g. `y ~ C(x)`
        would include only `y` and `x`). This may not always be possible for
        more advanced formulae that insert constants into the formula via the
        evaluation context rather than the data context.
        """

        variables: List[Variable] = [
            variable
            for term in self.__terms
            for factor in term.factors
            for variable in get_expression_variables(factor.expr, {})
            if "value" in variable.roles
        ]

        # Filter out constants like `contr` that are already present in the
        # TRANSFORMS namespace.
        from formulaic.transforms import TRANSFORMS

        return set(
            filter(
                lambda variable: variable.split(".", 1)[0] not in TRANSFORMS,
                Variable.union(variables),
            )
        )

    def __repr__(self) -> str:
        return " + ".join([str(t) for t in self.__terms])

    # Deprecated shims for legacy `Structured`-like behaviour (previously there
    # was no distinction between `SimpleFormula` and `StructuredFormula`, and
    # so it is possible that downstream libraries depend on these methods
    # existing).

    @property
    @deprecated(
        message="the `SimpleFormula.root` property is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def root(self) -> Self:
        return self

    @property
    @deprecated(
        message="the `SimpleFormula._has_root` property is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _has_root(self) -> bool:
        return True

    @property
    @deprecated(
        message="the `SimpleFormula._has_structure` property is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _has_structure(self) -> bool:
        return False

    @deprecated(
        message="the `SimpleFormula._map` method is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _map(
        self,
        func: Union[
            Callable[[SimpleFormula], Any],
            Callable[[SimpleFormula, Tuple[Union[str, int], ...]], Any],
        ],
        recurse: bool = True,
        as_type: Optional[Type[Structured]] = None,
        _context: Tuple[Union[str, int], ...] = (),
    ) -> Any:
        try:
            return func(self, ())  # type: ignore
        except TypeError:
            return func(self)  # type: ignore

    @deprecated(
        message="the `SimpleFormula._flatten` method is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _flatten(self) -> Generator[SimpleFormula, None, None]:
        yield self

    @deprecated(
        message="the `SimpleFormula._to_dict` method is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _to_dict(self) -> Dict[str, SimpleFormula]:
        return {"root": self}

    @deprecated(
        message="the `SimpleFormula._simplify` method is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _simplify(
        self, *, recurse: bool = True, unwrap: bool = True, inplace: bool = False
    ) -> SimpleFormula:
        return self

    @deprecated(
        message="the `SimpleFormula._update` method is deprecated.",
        as_of=(1, 1),
        removed_in=(2, 0),
    )
    def _update(self, root: Any = MISSING, **structure: Any) -> StructuredFormula:
        return StructuredFormula(
            root=self if root is MISSING else root, _ordering=self.ordering, **structure
        )


class StructuredFormula(Structured[SimpleFormula], Formula):
    """
    A container for structured formulae (formulae that have multiple nested
    components; for example left- and right-hand sides in a regression formula).

    This is a thin wrapper around `Structured[SimpleFormula]` that adds
    convenience methods for building model matrices from the formula (among
    other common tasks). You can build a `StructuredFormula` instance by
    directly instantiating it, or by using the `Formula` constructor.

    For more details on how to interact with structured formulae, refer to the
    `Structured` docs, or the formulaic user-guides.

    Attributes:
        _parser: The `FormulaParser` instance to use when parsing complete
            formulae (vs. individual terms). If not specified,
            `DefaultFormulaParser()` is used.
        _nested_parser: The `FormulaParser` instance to use when parsing
            strings describing nested or individual terms (e.g. when `spec` is a
            list of string term identifiers). If not specified and `_parser` is
            specified, `_parser` is used; if `_parser` is not specified,
            `DefaultFormulaParser(include_intercept=False)` is used instead.
        _ordering: The ordering method to apply to the terms implied by the
            formula specifications. Can be: "none", "degree" (default), or "sort".
    """

    __slots__ = ("_parser", "_nested_parser", "_ordering", "_context")

    def __init__(
        self,
        root: Union[FormulaSpec, MissingType] = MISSING,
        *,
        _ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
        _parser: Optional[FormulaParser] = None,
        _nested_parser: Optional[FormulaParser] = None,
        _context: Optional[Mapping[str, Any]] = None,
        **structure: FormulaSpec,
    ):
        self._ordering = OrderingMethod(_ordering)
        self._parser = _parser or DEFAULT_PARSER
        self._nested_parser = _nested_parser or _parser or DEFAULT_NESTED_PARSER
        self._context = _context
        super().__init__(root, **structure)  # type: ignore
        self._simplify(unwrap=False, inplace=True)

    def _prepare_item(  # type: ignore[override]
        self, key: str, item: FormulaSpec
    ) -> Union[SimpleFormula, StructuredFormula]:
        """
        Convert incoming formula items into either a `SimpleFormula` or a nested
        `StructuredFormula` instance.

        Note: Where parsing of strings is required, the nested-parser is used
        except for the root element of the parent formula.

        Args:
            key: The structural key where the item will be stored.
            item: The specification to convert.
        """
        if isinstance(item, Formula):
            return cast(Union[SimpleFormula, StructuredFormula], item)
        return Formula.from_spec(
            item,
            ordering=self._ordering,
            parser=(self._parser if key == "root" else self._nested_parser),
            nested_parser=self._nested_parser,
            context=self._context,
        )

    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        drop_rows: Optional[Set[int]] = None,
        **spec_overrides: Any,
    ) -> Union[ModelMatrix, Structured[ModelMatrix]]:
        """
        Build the model matrix (or matrices) realisation of this formula for the
        nominated `data`.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            drop_rows: An optional set of row indices to drop from the model
                matrix. If specified, it will also be updated during
                materialization with any additional rows dropped due to null
                values.
            spec_overrides: Any `ModelSpec` attributes to set/override. See
                `ModelSpec` for more details.
        """
        from .model_spec import ModelSpec

        return ModelSpec.from_spec(self, **spec_overrides).get_model_matrix(
            data, context=context, drop_rows=drop_rows
        )

    @property
    def required_variables(self) -> Set[Variable]:
        """
        The set of variables required in the data order to materialize this
        formula.

        Attempts are made to restrict these variables only to those expected in
        the data, and not, for example, those associated with transforms and/or
        values present in the evaluation namespace by default (e.g. `y ~ C(x)`
        would include only `y` and `x`). This may not always be possible for
        more advanced formulae that insert constants into the formula via the
        evaluation context rather than the data context.
        """

        variables: List[Variable] = []

        # Recurse through formula to collect all variables
        self._map(
            lambda formula: variables.extend(formula.required_variables),
        )

        return Variable.union(variables)

    def differentiate(  # pylint: disable=redefined-builtin
        self,
        *wrt: str,
        use_sympy: bool = False,
    ) -> SimpleFormula:
        """
        Take the gradient of this formula with respect to the variables in
        `wrt`.

        When used a linear regression context, making predictions based on the
        model matrices generated the differentiated formula is equivalent to
        estimating the gradient of the fitted model with respect to `wrt`.

        Args:
            wrt: The variables with respect to which the gradient should be
                taken.
            use_sympy: Whether to use sympy to perform symbolic differentiation.
        """
        return cast(
            SimpleFormula,
            self._map(lambda formula: formula.differentiate(*wrt, use_sympy=use_sympy)),
        )

    # Ensure pickling never includes context
    def __getstate__(self) -> Tuple[None, Dict[str, Any]]:
        slots = self.__slots__ + Structured.__slots__
        return (
            None,
            {
                slot: getattr(self, slot) if slot != "_context" else None
                for slot in slots
            },
        )
