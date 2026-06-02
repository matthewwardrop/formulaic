from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

from formulaic.utils.variables import Variable, get_required_variables

from .ordered_set import OrderedSet
from .term import Term

if TYPE_CHECKING:
    from .token import Token  # pragma: no cover


class Factor:
    """
    Factors are the indivisable atomic unit that make up formulas.

    Each instance of `Factor` is a specification that is evaluable by a
    materializer to generate concrete vector(s). `Factors` are multiplied
    together into `Term`s, which in turn represent the output columns of model
    matrices. Note that `Factor` instances are entirely abstract of data.

    Attributes:
        expr: The (string) expression to be evaluated by the materializer.
        eval_method: An `EvalMethod` enum instance indicating the mechanism to
            be used to evaluate the expression (one of: unknown, literal, lookup
            or python).
        kind: The kind of data represented (one of: unknown, constant,
            numerical, categorical).
        metadata: An additional (optional) dictionary of metadata (currently
            unused).
        token: The `Token` instance from which the the `Formula` object was
            created.
    """

    class EvalMethod(Enum):
        LITERAL = "literal"
        LOOKUP = "lookup"
        PYTHON = "python"

    class Kind(Enum):
        UNKNOWN = "unknown"
        CONSTANT = "constant"
        NUMERICAL = "numerical"
        CATEGORICAL = "categorical"

    __slots__ = ("expr", "_eval_method", "_kind", "metadata", "token")

    def __init__(
        self,
        expr: str = "",
        *,
        eval_method: Optional[Union[str, EvalMethod]] = None,
        kind: Optional[Union[str, Kind]] = None,
        metadata: Optional[dict] = None,
        token: Optional[Token] = None,
    ):
        self.expr = expr
        self.eval_method = eval_method  # type: ignore
        self.kind = kind  # type: ignore
        self.metadata = metadata or {}
        self.token = token

    @property
    def eval_method(self) -> EvalMethod:
        return self._eval_method

    @eval_method.setter
    def eval_method(self, eval_method: Union[str, EvalMethod]) -> None:
        self._eval_method = Factor.EvalMethod(eval_method or "lookup")

    @property
    def kind(self) -> Kind:
        return self._kind

    @kind.setter
    def kind(self, kind: Union[str, Factor.Kind]) -> None:
        self._kind = Factor.Kind(kind or "unknown")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.expr == other
        if isinstance(other, Factor):
            return self.expr == other.expr
        return NotImplemented

    def __hash__(self) -> int:
        return self.expr.__hash__()

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Factor):
            return self.expr < other.expr
        return NotImplemented

    def to_terms(
        self, *, context: Optional[Mapping[str, Any]] = None
    ) -> OrderedSet[Term]:
        """
        Convert this `Factor` instance into a `Term` instance, and expose it as
        a single-element ordered set.
        """
        return OrderedSet((Term([self]),))

    @property
    def required_variables(self) -> set[Variable]:
        """
        The set of variables required to evaluate this token.

        If this is a Python token, and the code is malformed and unable to be
        parsed, an empty set is returned. The code will fail more gracefully
        later on.

        Attempts are made to restrict these variables only to those expected in
        the data, and not, for example, those associated with transforms and/or
        values present in the evaluation namespace by default (e.g. `y ~ C(x)`
        would include only `y` and `x`). This may not always be possible for
        more advanced formulae that insert constants into the formula via the
        evaluation context rather than the data context. We also only consider
        the root variable (e.g. `a.fillna(0)` -> `a` vs. `a.fillna`).
        """
        if self.eval_method == Factor.EvalMethod.LOOKUP:
            return {Variable(self.expr, roles={Variable.Role.VALUE})}
        if self.eval_method == Factor.EvalMethod.PYTHON:
            from formulaic.transforms import TRANSFORMS

            try:
                # Filter out constants like `contr` that are already present in the
                # TRANSFORMS namespace.
                return set(
                    variable.root
                    for variable in get_required_variables(self.expr, TRANSFORMS)
                    if variable.root not in TRANSFORMS
                )
            except Exception:  # noqa: S110
                pass
        return set()

    def __repr__(self) -> str:
        if ":" in self.expr and self.eval_method == Factor.EvalMethod.LOOKUP:
            return f"`{self.expr}`"
        return self.expr
