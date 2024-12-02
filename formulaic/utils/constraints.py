from __future__ import annotations

import ast
import functools
import itertools
from numbers import Number
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy
from typing_extensions import Literal

from formulaic.parser.algos.tokenize import tokenize
from formulaic.parser.algos.tokens_to_ast import tokens_to_ast
from formulaic.parser.types import (
    ASTNode,
    Factor,
    Operator,
    OperatorResolver,
    Token,
)
from formulaic.parser.utils import exc_for_token

LinearConstraintSpec = Union[
    str,
    List[str],
    Dict[str, Number],
    Tuple["numpy.typing.ArrayLike", "numpy.typing.ArrayLike"],
    "numpy.typing.ArrayLike",
]


class LinearConstraints:
    """
    Represents linear constraints of form $Ax = b$, where $A$ is a matrix of
    coefficients for the features in $x$, and $b$ is a vector of constant
    values.

    Instances of this class are typically constructed via
    `ModelSpec.get_linear_constraints(...)` but can also be constructed
    directly for use in other contexts.

    Attributes:
        constraint_matrix: The matrix of coefficients on the features ($A$ from
            above). Each row is one constraint.
        constraint_values: The vector of constant values ($b$ from above).
        variable_names: The ordered names of the variables represented by $x$;
            typically the column names of a `ModelMatrix` instance.
    """

    @classmethod
    def from_spec(
        cls, spec: LinearConstraintSpec, variable_names: Optional[Sequence[str]] = None
    ) -> LinearConstraints:
        """
        Construct a `LinearConstraints` instance from a specification.

        Args:
            spec: The specification from which to derive the constraints. Can be
                a:
                    * str: In which case it is interpreted as a constraints
                        formula (e.g. "x + 2 * y = 3, z + y - x / 10"). All
                        variables used must be present in `variable_names`.
                    * List[str]: In which case the strings are joined with
                        commas and expected to look like `str` above.
                    * Dict[str, Number]: In which case each key is treated as
                        formula, and each value as the constraint (e.g. {"x":19}
                        , {"a + b": 0}).
                    * Tuple: a two-tuple describing the constraint matrix and
                        values respectively.
                    * numpy.ndarray/numerical sequence: a constraint matrix
                        (with all values assumed to be zero).
            variable_names: The ordered names of the variables represented by
                $x$; typically the column names of a `ModelMatrix` instance.
        """
        if isinstance(spec, LinearConstraints):
            return spec
        if (
            isinstance(spec, (str, dict))
            or isinstance(spec, list)
            and all(isinstance(s, str) for s in spec)
        ):
            if variable_names is None:
                raise ValueError(
                    "`variable_names` must be provided when parsing constraints from a formula."
                )
            if isinstance(spec, list):
                spec = ",".join(spec)
            if isinstance(spec, str):
                matrix, values = LinearConstraintParser(
                    variable_names=variable_names
                ).get_matrix(spec)
                return cls(matrix, values, variable_names)
            matrices, constants = [], []
            for key, constant in spec.items():
                matrix, values = LinearConstraintParser(
                    variable_names=variable_names
                ).get_matrix(key)
                matrices.append(matrix)
                constants.append(values + numpy.array(constant))
            return cls(
                numpy.vstack(matrices),
                numpy.hstack(constants),
                variable_names=variable_names,
            )
        if isinstance(spec, tuple) and len(spec) == 2:
            return cls(*spec, variable_names=variable_names)  # type: ignore
        return cls(spec, 0, variable_names=variable_names)  # type: ignore

    def __init__(
        self,
        constraint_matrix: "numpy.typing.ArrayLike",
        constraint_values: "numpy.typing.ArrayLike",
        variable_names: Optional[Sequence[str]] = None,
    ):
        """
        Attributes:
            constraint_matrix: The matrix of coefficients on the features ($A$ from
                above). Each row is one constraint.
            constraint_values: The vector of constant values ($b$ from above).
            variable_names: The ordered names of the variables represented by $x$;
                typically the column names of a `ModelMatrix` instance.
        """
        constraint_matrix = numpy.array(constraint_matrix)
        constraint_values = numpy.array(constraint_values)

        # Prepare incoming values
        if len(constraint_matrix.shape) == 1:
            constraint_matrix = constraint_matrix.reshape(1, *constraint_matrix.shape)
        if len(constraint_values.shape) == 0:
            constraint_values = constraint_values * numpy.ones(
                constraint_matrix.shape[0]
            )
        variable_names = variable_names or [
            f"x{i}" for i in range(constraint_matrix.shape[1])
        ]

        # Validate incoming values
        if len(constraint_matrix.shape) != 2:
            raise ValueError("`constraint_matrix` must be a 2D array.")
        if len(constraint_values.shape) != 1:
            raise ValueError("`constraint_values` must be a 1D array.")
        if constraint_values.shape[0] != constraint_matrix.shape[0]:
            raise ValueError(
                "Number of rows in constraint matrix does not equal the number of values in the values array."
            )
        if len(variable_names) != constraint_matrix.shape[1]:
            raise ValueError(
                "Number of column names does not match the number of columns in the linear constraint matrix."
            )

        self.constraint_matrix = constraint_matrix
        self.constraint_values = constraint_values
        self.variable_names = variable_names or [
            f"x{i}" for i in range(len(constraint_matrix))
        ]

    def __str__(self) -> str:
        out = []
        for i in range(self.constraint_matrix.shape[0]):
            out_one = []
            for nonzero_col in numpy.where(self.constraint_matrix[i, :])[0]:
                out_one.append(
                    f"{self.constraint_matrix[i, nonzero_col]} * {self.variable_names[nonzero_col]}"
                )
            out.append(" + ".join(out_one) + f" = {self.constraint_values[i]}")
        return "\n".join(out)

    def show(self) -> None:
        """
        Pretty-print the constraints.
        """
        print(str(self))

    @property
    def n_constraints(self) -> int:
        """
        The number of constraints represented by this `LinearConstraints`
        instance.
        """
        return self.constraint_matrix.shape[0]

    def __repr__(self) -> str:
        return f"<LinearConstraints: {self.n_constraints} constraints>"


class LinearConstraintParser:
    """
    A linear constraint parser.

    While this parser re-uses parts of the parser stack under `FormulaParser`,
    it interprets formulas using conventional algebra (rather than Wilkinson
    formulas).

    Attributes:
        variable_names: The ordered names of the variables for which constraints
            are being prepared. All variables used in the formula being parsed
            must be present in this sequence.
        operator_resolver: The operator resolver instance to use. If not
            provided, `ConstraintOperatorResolver` is used.
    """

    def __init__(
        self,
        variable_names: Sequence[str],
        operator_resolver: Optional[OperatorResolver] = None,
    ):
        self.variable_names = variable_names
        self.operator_resolver = operator_resolver or ConstraintOperatorResolver()

    def get_tokens(self, formula: str) -> Iterable[ConstraintToken]:
        """
        Tokenize a constraint formula.

        Args:
            formula: The constraint formula to tokenize.
        """
        return [ConstraintToken.for_token(token) for token in tokenize(formula)]

    def get_ast(self, formula: str) -> Optional[ASTNode]:
        """
        Assemble an abstract syntax tree for the nominated `formula` string.

        Args:
            formula: The constraint formula for which an AST should be
                generated.
        """
        return cast(
            Optional[ASTNode],
            tokens_to_ast(
                self.get_tokens(formula),
                operator_resolver=self.operator_resolver,
            ),
        )

    def get_terms(
        self, formula: str
    ) -> Union[None, List[ScaledFactor], Tuple[List[ScaledFactor], ...]]:
        """
        Build the `ScaledFactor` instances for a constraint formula string.

        Args:
            formula: The constraint formula for which to build terms.
        """
        ast = self.get_ast(formula)
        if not ast:
            return None
        return cast(
            Union[None, List[ScaledFactor], Tuple[List[ScaledFactor], ...]],
            ast.to_terms(),
        )

    def get_matrix(
        self, formula: str
    ) -> Tuple["numpy.typing.ArrayLike", "numpy.typing.ArrayLike"]:
        """
        Build the constraint matrix and constraint values vector associated with
        the parsed string.

        Args:
            formula: The constraint formula for which to build the constraint
                matrix and values vector.

        Returns:
            A tuple of the contraint matrix and constraint values respectively.
        """
        constraints = self.get_terms(formula)
        if not constraints:
            return numpy.empty((0, len(self.variable_names))), numpy.array([])

        if not isinstance(constraints, tuple):
            constraints = (constraints,)

        col_vectors = dict(
            zip(self.variable_names, numpy.eye(len(self.variable_names)))
        )

        matrix = []
        constants = []

        for constraint in constraints:
            vector = numpy.zeros(len(self.variable_names))
            constant: float = 0
            for scaled_factor in constraint:
                if scaled_factor.factor == 1:
                    constant += scaled_factor.scale
                else:
                    vector += (
                        scaled_factor.scale
                        * col_vectors[cast(Factor, scaled_factor.factor).expr]
                    )
            matrix.append(vector)
            constants.append(-constant)

        return numpy.array(matrix), numpy.array(constants)


class ConstraintToken(Token):
    """
    An enriched `Token` subclass that overrides `.to_terms()` to return
    a set of `ScaledFactor`s rather than `Terms`s.
    """

    @classmethod
    def for_token(cls, token: Token) -> ConstraintToken:
        return cls(
            **{
                attr: getattr(token, attr)
                for attr in ("token", "kind", "source", "source_start", "source_end")
            }
        )

    def to_terms(  # type: ignore[override]
        self, *, context: Optional[Mapping[str, Any]] = None
    ) -> Set[ScaledFactor]:
        if self.kind is Token.Kind.VALUE:
            factor = ast.literal_eval(self.token)
            if isinstance(factor, (int, float)):
                return {ScaledFactor(1, scale=factor)}
            raise exc_for_token(
                self,
                message="Only numeric literal values are permitted in constraint formulae.",
            )
        return {ScaledFactor(self.to_factor())}


class ScaledFactor:
    """
    A wrapper around a `Factor` instance that provides an additional "scale"
    attribute to allow storing information about the scalar coefficient of each
    `Factor`.

    Attributes:
        factor: The wrapped `Factor` instance.
        scale: The scalar value to be used as the coefficient of this factor.
    """

    def __init__(self, factor: Union[Factor, Literal[1]], *, scale: float = 1):
        self.factor = factor
        self.scale = scale

    def __add__(self, other: Any) -> ScaledFactor:
        if isinstance(other, ScaledFactor):
            return ScaledFactor(self.factor, scale=self.scale + other.scale)
        return NotImplemented  # pragma: no cover

    def __sub__(self, other: Any) -> ScaledFactor:
        if isinstance(other, ScaledFactor):
            return ScaledFactor(self.factor, scale=self.scale - other.scale)
        return NotImplemented  # pragma: no cover

    def __neg__(self) -> ScaledFactor:
        return ScaledFactor(self.factor, scale=-self.scale)

    def __hash__(self) -> int:
        return hash(self.factor)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ScaledFactor):
            return self.factor == other.factor
        return NotImplemented  # pragma: no cover

    def __repr__(self) -> str:
        return f"{self.scale}*{self.factor}"  # pragma: no cover


class ConstraintOperatorResolver(OperatorResolver):  # pylint: disable=unnecessary-lambda
    """
    The default constraint `OperatorResolver` implementation.

    These operators describe a regular algebra rather than a Wikinson formula
    one.
    """

    @property
    def operators(self) -> List[Operator]:
        def join_tuples(lhs: Any, rhs: Any) -> Tuple:
            if not isinstance(lhs, tuple):
                lhs = (lhs,)
            if not isinstance(rhs, tuple):
                rhs = (rhs,)
            return lhs + rhs

        def add_terms(
            terms_left: Set[ScaledFactor], terms_right: Set[ScaledFactor]
        ) -> Set[ScaledFactor]:
            terms_left = {term: term for term in terms_left}
            terms_right = {term: term for term in terms_right}

            added = set()

            for term in terms_left:
                if term in terms_right:
                    term = term + terms_right[term]
                added.add(term)
            added.update({term for term in terms_right if term not in added})

            return added

        def sub_terms(
            terms_left: Set[ScaledFactor], terms_right: Set[ScaledFactor]
        ) -> Set[ScaledFactor]:
            terms_left = {term: term for term in terms_left}
            terms_right = {term: term for term in terms_right}

            added = set()

            for term in terms_left:
                if term in terms_right:
                    term = term - terms_right[term]
                added.add(term)
            added.update(
                negate_terms({term for term in terms_right if term not in added})
            )

            return added

        def negate_terms(terms: Set[ScaledFactor]) -> Set[ScaledFactor]:
            return {-term for term in terms}

        def mul_terms(
            terms_left: Set[ScaledFactor], terms_right: Set[ScaledFactor]
        ) -> Set[ScaledFactor]:
            terms_left = {term: term for term in terms_left}
            terms_right = {term: term for term in terms_right}

            terms: Set[ScaledFactor] = set()

            for term_left, term_right in itertools.product(terms_left, terms_right):
                terms = add_terms(terms, {mul_term(term_left, term_right)})

            return terms

        def mul_term(term_left: ScaledFactor, term_right: ScaledFactor) -> ScaledFactor:
            if term_left.factor == 1:
                return ScaledFactor(
                    term_right.factor, scale=term_left.scale * term_right.scale
                )
            if term_right.factor == 1:
                return ScaledFactor(
                    term_left.factor, scale=term_left.scale * term_right.scale
                )
            raise RuntimeError(
                "Only one non-scalar factor can be involved in a linear constraint multiplication."
            )

        def div_terms(
            terms_left: Set[ScaledFactor], terms_right: Set[ScaledFactor]
        ) -> Set[ScaledFactor]:
            terms_left = {term: term for term in terms_left}
            terms_right = {term: term for term in terms_right}

            terms: Set[ScaledFactor] = set()

            for term_left, term_right in itertools.product(terms_left, terms_right):
                terms = add_terms(terms, {div_term(term_left, term_right)})

            return terms

        def div_term(term_left: ScaledFactor, term_right: ScaledFactor) -> ScaledFactor:
            if term_right.factor == 1:
                return ScaledFactor(
                    term_left.factor, scale=term_left.scale / term_right.scale
                )
            raise RuntimeError(
                "The right-hand operand must be a scalar in linear constraint division operations."
            )

        return [
            Operator(
                ",",
                arity=2,
                precedence=-200,
                associativity=None,
                to_terms=join_tuples,
                accepts_context=lambda context: all(
                    c.symbol == "," for c in context if isinstance(c, Operator)
                ),
                structural=True,
            ),
            Operator(
                "=",
                arity=2,
                precedence=-100,
                associativity=None,
                to_terms=lambda lhs, rhs: add_terms(lhs, negate_terms(rhs)),
            ),
            Operator(
                "+",
                arity=2,
                precedence=100,
                associativity="left",
                to_terms=lambda *args: functools.reduce(add_terms, args),
            ),
            Operator(
                "-",
                arity=2,
                precedence=100,
                associativity="left",
                to_terms=lambda left, right: sub_terms(left, right),
            ),
            Operator(
                "+",
                arity=1,
                precedence=100,
                associativity="right",
                fixity="prefix",
                to_terms=lambda arg: arg,
            ),
            Operator(
                "-",
                arity=1,
                precedence=100,
                associativity="right",
                fixity="prefix",
                to_terms=lambda arg: negate_terms(arg),
            ),
            Operator(
                "*",
                arity=2,
                precedence=200,
                associativity="left",
                to_terms=lambda lhs, rhs: mul_terms(lhs, rhs),
            ),
            Operator(
                "/",
                arity=2,
                precedence=200,
                associativity="left",
                to_terms=lambda lhs, rhs: div_terms(lhs, rhs),
            ),
        ]
