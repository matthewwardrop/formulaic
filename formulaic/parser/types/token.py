from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Optional, Union

from formulaic.utils.variables import Variable, get_expression_variables

from .factor import Factor
from .ordered_set import OrderedSet
from .term import Term


class Token:
    """
    The atomic unit into which formula strings are parsed.

    These tokens are intentionally very low-level, leaving interpretation and
    validation to higher-levels. As such, adding new operators/etc does not
    require any modification of this low-level code.

    The four kinds of token are:
      - context: a token used to scope terms into a given context
      - operator: an operator to be applied to other surrounding tokens (will
            always consist of non-word characters).
      - name: a name of a feature/variable to be lifted from the model matrix
            context.
      - value: a literal value (string/number).
      - python: a code string to be evaluated.

    Attributes:
        token: The portion of the formula string represented by this token.
        kind: The kind of this token (see above).
        source: The entire original source string.
        source_start: The index of the character within the string that starts
            this token.
        source_end: The index of the character within the string that ends
            this token.

    Note: These attributes *should* all be present, but may not be fully
    populated if generated outside of the default `tokenize()` implementation.
    """

    class Kind(Enum):
        CONTEXT = "context"
        OPERATOR = "operator"
        VALUE = "value"
        NAME = "name"
        PYTHON = "python"

    __slots__ = ("token", "_kind", "source", "source_start", "source_end")

    def __init__(
        self,
        token: str = "",
        *,
        kind: Optional[Union[str, Kind]] = None,
        source: Optional[str] = None,
        source_start: Optional[int] = None,
        source_end: Optional[int] = None,
    ):
        self.token = token
        self.kind = kind  # type: ignore
        self.source = source
        self.source_start = source_start
        self.source_end = source_end or source_start

    @property
    def kind(self) -> Optional[Kind]:
        return self._kind  # type: ignore

    @kind.setter
    def kind(self, kind: Optional[Union[str, Kind]]) -> None:
        self._kind = self.Kind(kind) if kind else kind

    def update(
        self, char: str, source_index: int, kind: Union[None, str, Kind] = None
    ) -> Token:
        """
        Add a character to the token string, keeping track of the source
        indices.

        Args:
            char: The character to add.
            source_index: The index of the character within the source string.
            kind: If present, the kind of the token is updated to reflect the
                nominated kind.

        Returns:
            A reference to this token instance.
        """
        self.token += char
        if self.source_start is None:
            self.source_start = source_index
        self.source_end = source_index
        if kind is not None:
            self.kind = Token.Kind(kind)
        return self

    def __bool__(self) -> bool:
        return bool(self.token)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.token == other
        if isinstance(other, Token):
            return self.token == other.token and self.kind == other.kind
        return NotImplemented

    def __hash__(self) -> int:
        return self.token.__hash__()

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Token):
            return self.token < other.token
        return NotImplemented

    @property
    def source_loc(self) -> tuple[Optional[int], Optional[int]]:
        """
        The indices of the first and last character represented by this token in
        the source string.
        """
        return (self.source_start, self.source_end)

    def to_factor(self) -> Factor:
        """
        A `Factor` instance corresponding to this token. Note that operator
        tokens cannot be converted to tokens.
        """
        if self.kind is None:  # pragma: no cover
            raise RuntimeError("`Token.kind` has not been set.")
        kind_to_eval_method = {
            Token.Kind.NAME: "lookup",
            Token.Kind.PYTHON: "python",
            Token.Kind.VALUE: "literal",
        }
        return Factor(
            expr=self.token,
            eval_method=kind_to_eval_method[self.kind],
            token=self,
        )

    def to_terms(
        self, *, context: Optional[Mapping[str, Any]] = None
    ) -> OrderedSet[Term]:
        """
        An order set of `Term` instances for this token. This will just be
        an iterable with one `Term` having one `Factor` (that generated by
        `.to_factor()`). Operator tokens cannot be converted to an iterable of
        `Term`s.
        """
        return OrderedSet((Term([self.to_factor()]),))

    def flatten(self, str_args: bool = False) -> Any:
        """
        Return this token (or if `str_args` is `True`, a string representation
        of this token).

        Args:
            str_args: Whether to convert this token to a string during
            flattening.
        """
        return str(self) if str_args else self

    def get_source_context(self, colorize: bool = False) -> Optional[str]:
        """
        Render a string that highlights the location of this token in the source
        string.

        Args:
            colorize: Whether to highlight the location of this token in bold
                red font.
        """
        if not self.source or self.source_start is None or self.source_end is None:
            return None
        if colorize:
            RED_BOLD = "\x1b[1;31m"
            RESET = "\x1b[0m"
            return f"{self.source[:self.source_start]}⧛{RED_BOLD}{self.source[self.source_start:self.source_end+1]}{RESET}⧚{self.source[self.source_end+1:]}"
        return f"{self.source[:self.source_start]}⧛{self.source[self.source_start:self.source_end+1]}⧚{self.source[self.source_end+1:]}"

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
        evaluation context rather than the data context.
        """
        if self.kind is Token.Kind.NAME:
            return {Variable(self.token)}
        if self.kind is Token.Kind.PYTHON:
            try:
                # Filter out constants like `contr` that are already present in the
                # TRANSFORMS namespace.
                from formulaic.transforms import TRANSFORMS

                return set(
                    filter(
                        lambda variable: variable.split(".", 1)[0] not in TRANSFORMS,
                        get_expression_variables(self.token),
                    )
                )
            except Exception:  # noqa: S110
                pass
        return set()

    def __repr__(self) -> str:
        return self.token

    # Additional methods for later mutation

    def copy_with_attrs(self, **attrs: Any) -> Token:
        """
        Return a copy of this `Token` instance with attributes set from attrs.

        Args:
            attrs: Attribute keys and values to set on the copy of this
                instance.
        """
        new_token = copy.copy(self)
        for attr, value in attrs.items():
            setattr(new_token, attr, value)
        return new_token

    def split(
        self, pattern: Union[str, re.Pattern], after: bool = False, before: bool = False
    ) -> Iterable[Token]:
        """
        Split this instance into multple tokens around all non-overlapping
        matches of `pattern`.

        Args:
            pattern: The pattern by which to split this `Token` instance.
            after: Whether to split after the pattern.
            before: Whether to split before the pattern.
        """
        if not after and not before:
            yield self
            return

        if not isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)

        last_index = 0
        separators = pattern.finditer(self.token)

        def get_next_token(next_index: int) -> tuple[int, Token]:
            return next_index, self.copy_with_attrs(
                token=self.token[last_index:next_index]
            )

        for separator in separators:
            if before:
                last_index, new_token = get_next_token(separator.span()[0])
                yield new_token
            if after:
                last_index, new_token = get_next_token(separator.span()[1])
                yield new_token

        if last_index < len(self.token):
            yield get_next_token(len(self.token))[1]
