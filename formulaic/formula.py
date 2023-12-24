from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, cast

from typing_extensions import TypeAlias

from .errors import FormulaInvalidError
from .model_matrix import ModelMatrix
from .parser import DefaultFormulaParser
from .parser.types import FormulaParser, OrderedSet, Structured, Term
from .utils.calculus import differentiate_term


FormulaSpec: TypeAlias = Union[
    str,
    List[Union[str, Term]],
    Set[Union[str, Term]],
    Structured[Union[str, List[Term], Set[Term]]],
    "Formula",  # Direct formula specification
    Dict[str, "FormulaSpec"],
    Tuple["FormulaSpec", ...],  # Structured formulae
]


class OrderingMethod(Enum):
    NONE = "none"
    DEGREE = "degree"
    SORT = "sort"


class Formula(Structured[List[Term]]):
    """
    A Formula is a (potentially structured) list of terms, which is represented
    by this class.

    This is a thin wrapper around `Strucuted[List[Term]]` that adds convenience
    methods for building model matrices from the formula (among other common
    tasks). You can build a `Formula` instance by passing in a string for
    parsing, or by manually assembling the terms yourself.

    Examples:
    ```
    >>> Formula("y ~ x")
    .lhs:
        y
    .rhs:
        1 + x
    >>> Formula("x + y", a=["x", "y:z"], b="y ~ z")
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

    You can control how strings are parsed into terms by passing in custom
    parsers via `_parser` and `_nested_parser`.
    ```
    >>> Formula("y ~ x", _parser=DefaultFormulaParser(include_intercept=False))
    .lhs:
        y
    .rhs:
        x
    ```

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
            formula `spec`. Can be: "none", "degree" (default), or "sort".
    """

    DEFAULT_PARSER = DefaultFormulaParser()
    DEFAULT_NESTED_PARSER = DefaultFormulaParser(include_intercept=False)

    __slots__ = ("_parser", "_nested_parser", "_ordering")

    @classmethod
    def from_spec(
        cls,
        spec: FormulaSpec,
        *,
        parser: Optional[FormulaParser] = None,
        nested_parser: Optional[FormulaParser] = None,
        ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
    ) -> Formula:
        """
        Construct a `Formula` instance from a formula specification.

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
            return spec
        return Formula(
            spec, _parser=parser, _nested_parser=nested_parser, _ordering=ordering
        )

    def __init__(
        self,
        *args: FormulaSpec,
        _parser: Optional[FormulaParser] = None,
        _nested_parser: Optional[FormulaParser] = None,
        _ordering: Union[OrderingMethod, str] = OrderingMethod.DEGREE,
        **kwargs: Any,
    ):
        self._parser = _parser or self.DEFAULT_PARSER
        self._nested_parser = _nested_parser or _parser or self.DEFAULT_NESTED_PARSER
        self._ordering = OrderingMethod(_ordering)
        super().__init__(*args, **kwargs)
        self._simplify(unwrap=False, inplace=True)

    def _prepare_item(self, key: str, item: FormulaSpec) -> Union[List[Term], Formula]:  # type: ignore[override]
        """
        Convert incoming formula items into either a list of Terms or a nested
        `Formula` instance.

        Note: Where parsing of strings is required, the nested-parser is used
        except for the root element of the parent formula.

        Args:
            key: The structural key where the item will be stored.
            item: The specification to convert.
        """

        if isinstance(item, str):
            item = cast(
                FormulaSpec,
                (self._parser if key == "root" else self._nested_parser)
                .get_terms(item)
                ._simplify(),
            )

        if isinstance(item, Structured):
            formula_or_terms = Formula(
                _parser=self._nested_parser, **item._structure
            )._simplify()
        elif isinstance(item, (list, set, OrderedSet)):
            formula_or_terms = [
                term
                for value in item
                for term in (
                    self._nested_parser.get_terms(value)  # type: ignore[attr-defined]
                    if isinstance(value, str)
                    else [value]
                )
            ]
            self.__validate_terms(formula_or_terms)
        else:
            raise FormulaInvalidError(
                f"Unrecognized formula specification: {repr(item)}."
            )

        # Order terms appropriately
        orderer = None
        if self._ordering is OrderingMethod.DEGREE:
            orderer = lambda terms: sorted(terms, key=lambda term: term.degree)
        elif self._ordering is OrderingMethod.SORT:
            orderer = lambda terms: sorted(
                [Term(factors=sorted(term.factors)) for term in terms]
            )

        if orderer is not None:
            if isinstance(formula_or_terms, Structured):
                formula_or_terms = formula_or_terms._map(orderer)
            else:
                formula_or_terms = orderer(formula_or_terms)

        return cast(Union[List[Term], Formula], formula_or_terms)

    @classmethod
    def __validate_terms(cls, formula_or_terms: Any) -> None:
        """
        Verify that all terms are of the appropriate type. The acceptable types
        are:
            - List[Terms]
            - Tuple[List[Terms], ...]
            - Formula
        """
        if not isinstance(formula_or_terms, list):
            # Should be impossible to reach this; here as a sentinel
            raise FormulaInvalidError(
                f"All components of a formula should be lists of `Term` instances. Found: {repr(formula_or_terms)}."
            )
        for term in formula_or_terms:
            if not isinstance(term, Term):
                raise FormulaInvalidError(
                    f"All terms in formula should be instances of `formulaic.parser.types.Term`; received term {repr(term)} of type `{type(term)}`."
                )

    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        **spec_overrides: Any,
    ) -> Union[ModelMatrix, Structured[ModelMatrix]]:
        """
        Build the model matrix (or matrices) realisation of this formula for the
        nominated `data`.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            spec_overrides: Any `ModelSpec` attributes to set/override. See
                `ModelSpec` for more details.
        """
        from .model_spec import ModelSpec

        return ModelSpec.from_spec(self, **spec_overrides).get_model_matrix(
            data, context=context
        )

    def differentiate(  # pylint: disable=redefined-builtin
        self,
        *wrt: str,
        use_sympy: bool = False,
    ) -> Formula:
        """
        EXPERIMENTAL: Take the gradient of this formula. When used a linear
        regression, evaluating a trained model on model matrices generated by
        this formula is equivalent to estimating the gradient of that fitted
        form with respect to `wrt`.

        Args:
            wrt: The variables with respect to which the gradient should be
                taken.
            use_sympy: Whether to use sympy to perform symbolic differentiation.


        Notes:
            This method is provisional and may be removed in any future major
            version.
        """
        return cast(
            Formula,
            self._map(
                lambda terms: [
                    differentiate_term(term, wrt, use_sympy=use_sympy) for term in terms
                ]
            ),
        )

    def __getattr__(self, attr: str) -> Any:
        # Keep substructures wrapped to retain access to helper functions.
        subformula = super().__getattr__(attr)
        if attr != "root":
            return Formula.from_spec(subformula)
        return subformula

    def __getitem__(self, key: Any) -> Any:
        # Keep substructures wrapped to retain access to helper functions.
        subformula = super().__getitem__(key)
        if key != "root":
            return Formula.from_spec(subformula)
        return subformula

    def __repr__(self, to_str: Callable[..., str] = repr) -> str:
        if not self._has_structure and self._has_root:
            return " + ".join([str(t) for t in self])
        return str(self._map(lambda terms: " + ".join([str(t) for t in terms])))
