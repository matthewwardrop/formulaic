from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from typing_extensions import TypeAlias

from .errors import FormulaInvalidError, FormulaMaterializerInvalidError
from .materializers.base import FormulaMaterializer
from .model_matrix import ModelMatrix
from .parser import DefaultFormulaParser
from .parser.types import FormulaParser, Structured, Term
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


class Formula:
    """
    A representation of a "formula".

    A formula is basically just a (structured) set of terms to include in the
    model matrixes, with implicit or explicit encoding choices.

    Attributes:
        terms: The terms defined by the (parsed) formula.
    """

    @classmethod
    def from_spec(
        cls,
        spec: FormulaSpec,
        parser: Optional[FormulaParser] = None,
        term_parser: Optional[FormulaParser] = None,
    ) -> Formula:
        """
        Construct a `Formula` instance from a formula specification.

        Args:
            spec: The formula specification.
            parser: The `FormulaParser` instance to use when parsing complete
                formulae (vs. individual terms). If not specified,
                `DefaultFormulaParser()` is used.
            term_parser: The `FormulaParser` instance to use when parsing
                strings describing individual terms (e.g. when `spec` is a
                list of string term identifiers). If not specified and `parser`
                is specified, `parser` is used; if `parser` is not specified,
                `DefaultFormulaParser(include_intercept=False)` is used instead.
        """
        if isinstance(spec, Formula):
            return spec
        return cls(spec, parser=parser, term_parser=term_parser)

    terms: Structured[List[Term]]

    def __init__(
        self,
        spec: FormulaSpec,
        parser: Optional[FormulaParser] = None,
        term_parser: Optional[FormulaParser] = None,
    ):
        if isinstance(spec, str):
            parser = parser or DefaultFormulaParser()
            terms = parser.get_terms(spec, sort=True)
        elif isinstance(spec, Formula):
            terms = spec.terms
        elif isinstance(spec, (list, set)):
            term_parser = (
                term_parser or parser or DefaultFormulaParser(include_intercept=False)
            )
            terms = [
                term
                for value in spec
                for term in (
                    term_parser.get_terms(value) if isinstance(value, str) else [value]
                )
            ]
        elif isinstance(spec, tuple):
            terms = tuple(
                Formula.from_spec(group, parser=parser, term_parser=term_parser).terms
                for group in spec
            )
        elif isinstance(spec, Structured):
            term_parser = (
                term_parser or parser or DefaultFormulaParser(include_intercept=False)
            )
            terms = spec._map(
                lambda nested_spec: [
                    term
                    for value in nested_spec
                    for term in (
                        term_parser.get_terms(value)
                        if isinstance(value, str)
                        else [value]
                    )
                ]
            )
        else:
            raise FormulaInvalidError(
                f"Unrecognized formula specification: {repr(spec)}."
            )
        self.terms = terms

    @property
    def terms(self) -> Structured[Term]:
        """
        The terms associated with this formula.
        """
        return self._terms

    @terms.setter
    def terms(self, terms: Union[List[Term], Set[Term], Structured[List[Term]]]):
        if not isinstance(terms, Structured):
            terms = Structured(terms)
        self.__check_terms(terms)
        self._terms = terms

    @classmethod
    def __check_terms(cls, terms):
        if isinstance(terms, Structured):
            return terms._map(cls.__check_terms)
        if not isinstance(terms, list):
            # Should be impossible to reach this; here as a sentinel
            raise FormulaInvalidError(
                f"All components of a formula should be lists of `Term` instances. Found: {repr(terms)}."
            )
        for term in terms:
            if not isinstance(term, Term):
                raise FormulaInvalidError(
                    f"All terms in formula should be instances of `formulaic.parser.types.Term`; received term {repr(term)} of type `{type(term)}`."
                )

    def get_model_matrix(
        self,
        data: Any,
        context: Optional[Mapping[str, Any]] = None,
        materializer: Optional[FormulaMaterializer] = None,
        ensure_full_rank: bool = True,
        **kwargs,
    ) -> Union[ModelMatrix, Structured[ModelMatrix]]:
        """
        Build the model matrix (or matrices) realisation of this formula for the
        nominated `data`.

        Args:
            data: The data for which to build the model matrices.
            context: An additional mapping object of names to make available in
                when evaluating formula term factors.
            materializer: The `FormulaMatericalizer` class to use when
                materializing the data. If not specified, an attempt is made to
                automatically detect this based on the type of `data` (e.g.
                pandas DataFrames |-> `PandasMaterializer`).
            ensure_full_rank: Whether to ensure the model matrices are
                structurally full rank (contain no columns that are guaranteed
                to be linearly dependent).
            kwargs: Additional materializer-specific arguments to pass on to the
                materializer's `.get_model_matrix` method.
        """
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

    def differentiate(self, *vars: Tuple[str, ...], use_sympy: bool = False):
        """
        EXPERIMENTAL: Take the gradient of this formula. When used a linear
        regression, evaluating a trained model on model matrices generated by
        this formula is equivalent to estimating the gradient of that fitted
        form with respect to `vars`.

        Args:
            vars: The variables with respect to which the gradient should be
                taken.
            use_sympy: Whether to use sympy to perform symbolic differentiation.


        Notes:
            This method is provisional and may be removed in any future major
            version.
        """
        return Formula(
            self.terms._map(
                lambda terms: [
                    differentiate_term(term, vars, use_sympy=use_sympy)
                    for term in terms
                ]
            )
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
        if (
            self.terms._has_structure
            or self.terms._has_root
            and isinstance(self.terms.root, tuple)
        ):
            return str(
                self.terms._map(lambda terms: " + ".join(str(term) for term in terms))
            )
        return " + ".join(str(term) for term in self.terms)

    def __eq__(self, other):
        if isinstance(other, Formula):
            return self.terms == other.terms
        return NotImplemented

    def __repr__(self):
        return str(self)
