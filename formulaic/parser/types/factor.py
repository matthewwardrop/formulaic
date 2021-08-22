from enum import Enum
from typing import Dict, Iterable, Optional, Union

from .term import Term


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
    """

    class EvalMethod(Enum):
        UNKNOWN = "unknown"
        LITERAL = "literal"
        LOOKUP = "lookup"
        PYTHON = "python"

    class Kind(Enum):
        UNKNOWN = "unknown"
        CONSTANT = "constant"
        NUMERICAL = "numerical"
        CATEGORICAL = "categorical"

    __slots__ = ("expr", "_eval_method", "_kind", "metadata")

    def __init__(
        self,
        expr: str = "",
        *,
        eval_method: Optional[Union[str, EvalMethod]] = None,
        kind: Optional[Union[str, Kind]] = None,
        metadata: Optional[Dict] = None
    ):
        self.expr = expr
        self.eval_method = eval_method
        self.kind = kind
        self.metadata = metadata or {}

    @property
    def eval_method(self) -> EvalMethod:
        return self._eval_method

    @eval_method.setter
    def eval_method(self, eval_method):
        self._eval_method = Factor.EvalMethod(eval_method or "unknown")

    @property
    def kind(self) -> Kind:
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = Factor.Kind(kind or "unknown")

    def __eq__(self, other):
        if isinstance(other, str):
            return self.expr == other
        if isinstance(other, Factor):
            return self.expr == other.expr
        return NotImplemented

    def __hash__(self):
        return self.expr.__hash__()

    def __lt__(self, other):
        if isinstance(other, Factor):
            return self.expr < other.expr
        return NotImplemented

    def to_terms(self) -> Iterable[Term]:
        """
        Convert this `Factor` instance into a `Term` instance, and expose it as
        a single-element iterable.
        """
        return {Term([self])}

    def __repr__(self):
        return self.expr
