from __future__ import annotations

import ast
import functools
import itertools
import re
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Iterable, List, Sequence, Set, Tuple, Union, cast

from typing_extensions import Self

from .algos.sanitize_tokens import sanitize_tokens
from .algos.tokenize import tokenize
from .types import (
    Factor,
    FormulaParser,
    Operator,
    OperatorResolver,
    OrderedSet,
    Structured,
    Term,
    Token,
)
from .utils import (
    exc_for_token,
    insert_tokens_after,
    merge_operator_tokens,
    replace_tokens,
)


class DefaultParserFeatureFlag(Flag):
    """
    Feature flags to restrict the flexibility of the formula parser.
    """

    TWOSIDED = auto()
    MULTIPART = auto()
    MULTISTAGE = auto()

    # Convenience flags
    NONE = 0
    DEFAULT = TWOSIDED | MULTIPART
    ALL = TWOSIDED | MULTIPART | MULTISTAGE

    @classmethod
    def from_spec(
        cls, flags: Union[DefaultParserFeatureFlag, Set[str]]
    ) -> DefaultParserFeatureFlag:
        if isinstance(flags, DefaultParserFeatureFlag):
            return flags
        result = cls.NONE
        for flag in flags:
            result |= getattr(cls, flag.upper())
        return result


@dataclass
class DefaultFormulaParser(FormulaParser):
    """
    The default parser for `Formula`s.

    It extends `FormulaParser` by defaulting the operator resolver to
    `DefaultOperatorResolver`, and by adding the option to enable the inclusion
    of an intercept.

    Attributes:
        operator_resolver: The operator resolver to use when parsing the formula
            string and generating the abstract syntax tree. If not specified,
            it will default to `DefaultOperatorResolver`.
        include_intercept: Whether to include an intercept by default
                (formulas can still omit this intercept in the usual manner:
                adding a '-1' or '+0' term).
        feature_flags: Feature flags to enable or disable certain features. Can
            be passed in as a `DefaultParserFeatureFlag` value or as a set of string flags
            (which will be cast to a `DefaultParserFeatureFlag` instance internally).
    """

    ZERO_PATTERN = re.compile(r"(?:^|(?<=\W))0(?=\W|$)")

    # Attributes
    operator_resolver: OperatorResolver = field(
        default_factory=lambda: DefaultOperatorResolver()  # pylint: disable=unnecessary-lambda
    )
    include_intercept: bool = True
    feature_flags: DefaultParserFeatureFlag = DefaultParserFeatureFlag.DEFAULT

    def __post_init__(self) -> None:
        if isinstance(self.feature_flags, set):
            self.feature_flags = DefaultParserFeatureFlag.from_spec(self.feature_flags)
        if isinstance(self.operator_resolver, DefaultOperatorResolver):
            self.operator_resolver.set_feature_flags(self.feature_flags)

    def set_feature_flags(self, flags: DefaultParserFeatureFlag | Set[str]) -> Self:
        self.feature_flags = DefaultParserFeatureFlag.from_spec(flags)
        self.__post_init__()
        return self

    def get_tokens(self, formula: str) -> Iterable[Token]:
        """
        Return an iterable of `Token` instances for the nominated `formula`
        string.

        Args:
            formula: The formula string to be tokenized.
        """

        # Transform formula to add intercepts and replace 0 with -1. We do this
        # as token transformations to reduce the complexity of the code, and
        # also to avoid the ambiguity in the AST around intentionally unary vs.
        # incidentally unary operations (e.g. "+0" vs. "x + (+0)"). This cannot
        # easily be done as string operations because of quotations and escapes
        # which are best left to the tokenizer.

        token_one = Token("1", kind=Token.Kind.VALUE)
        token_plus = Token("+", kind=Token.Kind.OPERATOR)
        token_minus = Token("-", kind=Token.Kind.OPERATOR)

        tokens = sanitize_tokens(tokenize(formula))

        # Substitute "0" with "-1"
        tokens = replace_tokens(
            tokens, "0", [token_minus, token_one], kind=Token.Kind.VALUE
        )

        # Insert intercepts
        if self.include_intercept:
            tokens = list(
                insert_tokens_after(
                    tokens,
                    "~",
                    [token_one],
                    kind=Token.Kind.OPERATOR,
                    join_operator="+",
                )
            )

            def find_rhs_index(tokens: List[Token]) -> int:
                """
                Find the top-level index of the tilde operator starting the
                right hand side of the formula (or -1 if not found).
                """
                from .algos.tokens_to_ast import CONTEXT_CLOSERS, CONTEXT_OPENERS

                context = []
                for index, token in enumerate(tokens):
                    if token.kind is Token.Kind.CONTEXT:
                        if token.token in CONTEXT_OPENERS:
                            context.append(token.token)
                            continue
                        else:
                            if (
                                not context
                                or context[-1] != CONTEXT_CLOSERS[token.token]
                            ):
                                return -1  # pragma: no cover ; should not happen
                            context.pop()
                    if context:
                        continue
                    if token.token == "~":  # noqa: S105
                        return index
                return -1

            rhs_index = find_rhs_index(tokens) + 1
            tokens = [
                *(
                    tokens[:rhs_index]
                    if rhs_index > 0
                    else ([token_one, token_plus] if len(tokens) > 0 else [token_one])
                ),
                *insert_tokens_after(
                    tokens[rhs_index:],
                    r"\|",
                    [token_one],
                    kind=Token.Kind.OPERATOR,
                    join_operator="+",
                ),
            ]

        # Collapse inserted "+" and "-" operators to prevent unary issues.
        tokens = merge_operator_tokens(tokens, symbols={"+", "-"})

        return tokens

    def get_terms(self, formula: str) -> Structured[List[Term]]:
        """
        Assemble the `Term` instances for a formula string. Depending on the
        operators involved, this may be an iterable of `Term` instances, or
        an iterable of iterables of `Term`s, etc.

        This implementation also verifies that the formula is well-formed, in
        that it does not have any literals apart from 1 or numeric scaling of
        other terms.

        Args:
            formula: The formula for which an AST should be generated.
        """
        terms = super().get_terms(formula)

        def check_terms(terms: Iterable[Term]) -> None:
            seen_terms = set()
            for term in terms:
                if len(term.factors) == 1:
                    factor = term.factors[0]
                    if (
                        factor.eval_method is Factor.EvalMethod.LITERAL
                        and factor.expr != "1"
                    ):
                        raise exc_for_token(
                            factor.token or Token(),
                            "Numeric literals other than `1` can only be used "
                            "to scale other terms. (tip: Use `:` rather than "
                            "`*` when scaling terms)"
                            if factor.expr.replace(".", "", 1).isnumeric()
                            else "String literals are not valid in formulae.",
                        )
                else:
                    for factor in term.factors:
                        if (
                            factor.eval_method is Factor.EvalMethod.LITERAL
                            and not factor.expr.replace(".", "", 1).isnumeric()
                        ):
                            raise exc_for_token(
                                factor.token or Token(),
                                "String literals are not valid in formulae.",
                            )

                term_hash = tuple(
                    factor.expr
                    for factor in term.factors
                    if factor.eval_method != Factor.EvalMethod.LITERAL
                )
                if term_hash in seen_terms:
                    raise exc_for_token(
                        term.factors[0].token or Token(),
                        "Term already seen with a different numerical scaling. "
                        "(tip: Use `:` rather than `*` when scaling terms)",
                    )
                seen_terms.add(term_hash)

        terms._map(check_terms)

        return terms


@dataclass
class DefaultOperatorResolver(OperatorResolver):
    """
    The default operator resolver implementation.

    This class implements the standard operators in a form consistent with
    other implementations of Wilkinson formulas. It can be extended via
    subclassing to support other kinds of operators, in which case `.operators`
    and/or `.resolve` can be overridden. For more details about which operators
    are implemented, review the code or the documentation website.

    Attributes:
        feature_flags: Feature flags to enable or disable certain features. Can
            be passed in as a `DefaultParserFeatureFlag` value or as a set of string
            flags (which will be cast to a `DefaultParserFeatureFlag` instance
            internally).
    """

    feature_flags: DefaultParserFeatureFlag = DefaultParserFeatureFlag.DEFAULT

    def __post_init__(self) -> None:
        if isinstance(self.feature_flags, set):
            self.feature_flags = DefaultParserFeatureFlag.from_spec(self.feature_flags)

    def set_feature_flags(self, flags: DefaultParserFeatureFlag | Set[str]) -> Self:
        self.feature_flags = DefaultParserFeatureFlag.from_spec(flags)
        if "operator_table" in self.__dict__:
            del self.__dict__["operator_table"]
        return self

    @property
    def operators(self) -> List[Operator]:
        def formula_part_expansion(
            lhs: OrderedSet[Term], rhs: OrderedSet[Term]
        ) -> Tuple[OrderedSet[Term], ...]:
            terms = (lhs, rhs)

            out = []
            for termset in terms:
                if isinstance(termset, tuple):
                    out.extend(termset)
                else:
                    out.append(termset)
            return tuple(out)

        def nested_product_expansion(
            parents: OrderedSet[Term], nested: OrderedSet[Term]
        ) -> OrderedSet[Term]:
            common = functools.reduce(lambda x, y: x * y, parents)
            return cast(
                OrderedSet, parents | OrderedSet(common * term for term in nested)
            )

        def power(arg: OrderedSet[Term], power: OrderedSet[Term]) -> OrderedSet[Term]:
            power_term = next(iter(power))
            if (
                not len(power_term.factors) == 1
                or not power_term.factors[0].token
                or power_term.factors[0].token.kind is not Token.Kind.VALUE
                or not isinstance(ast.literal_eval(power_term.factors[0].expr), int)
            ):
                raise exc_for_token(
                    power_term.factors[0].token or Token(),
                    "The right-hand argument of `**` must be a positive integer.",
                )
            return OrderedSet(
                functools.reduce(lambda x, y: x * y, term)
                for term in itertools.product(*[arg] * int(power_term.factors[0].expr))
            )

        def multistage_formula(
            lhs: OrderedSet[Term], rhs: OrderedSet[Term]
        ) -> Structured[OrderedSet[Term]]:
            def get_terms(terms: OrderedSet[Term]) -> List[Term]:
                return [
                    Term(
                        factors=[Factor(str(t) + "_hat", eval_method="lookup")],
                        origin=t,
                    )
                    for t in terms
                ]

            if isinstance(lhs, Structured):
                raise NotImplementedError(
                    "Nested multistage formulas do not support structured lhs."
                )

            return Structured(get_terms(lhs), deps=(Structured(lhs=lhs, rhs=rhs),))

        return [
            Operator(
                "~",
                arity=2,
                precedence=-100,
                associativity=None,
                to_terms=lambda lhs, rhs: Structured(lhs=lhs, rhs=rhs),
                accepts_context=lambda context: len(context) == 0,
                structural=True,
                disabled=DefaultParserFeatureFlag.TWOSIDED not in self.feature_flags,
            ),
            Operator(
                "~",
                arity=1,
                precedence=-100,
                associativity=None,
                fixity="prefix",
                to_terms=lambda terms: terms,
                accepts_context=lambda context: len(context) == 0,
                structural=True,
            ),
            Operator(
                "~",
                arity=2,
                precedence=-100,
                associativity=None,
                to_terms=multistage_formula,
                accepts_context=lambda context: bool(context) and context[-1] == "[",
                structural=True,
                disabled=DefaultParserFeatureFlag.MULTISTAGE not in self.feature_flags,
            ),
            Operator(
                "|",
                arity=2,
                precedence=-50,
                associativity=None,
                to_terms=formula_part_expansion,
                accepts_context=lambda context: all(
                    isinstance(c, Operator) and c.symbol in "~|" for c in context
                ),
                structural=True,
                disabled=DefaultParserFeatureFlag.MULTIPART not in self.feature_flags,
            ),
            Operator(
                "+",
                arity=2,
                precedence=100,
                associativity="left",
                to_terms=lambda lhs, rhs: lhs | rhs,
            ),
            Operator(
                "-",
                arity=2,
                precedence=100,
                associativity="left",
                to_terms=lambda left, right: left - right,
            ),
            Operator(
                "+",
                arity=1,
                precedence=100,
                associativity="right",
                fixity="prefix",
                to_terms=lambda terms: terms,
            ),
            Operator(
                "-",
                arity=1,
                precedence=100,
                associativity="right",
                fixity="prefix",
                to_terms=lambda terms: OrderedSet(),
            ),
            Operator(
                "*",
                arity=2,
                precedence=200,
                associativity="left",
                to_terms=lambda *term_sets: (
                    OrderedSet(itertools.chain(*term_sets))
                    | OrderedSet(
                        functools.reduce(lambda x, y: x * y, term)
                        for term in itertools.product(*term_sets)
                    )
                ),
            ),
            Operator(
                "/",
                arity=2,
                precedence=200,
                associativity="left",
                to_terms=nested_product_expansion,
            ),
            Operator(
                "in",
                arity=2,
                precedence=200,
                associativity="left",
                to_terms=lambda nested, parents: nested_product_expansion(
                    parents, nested
                ),
            ),
            Operator(
                ":",
                arity=2,
                precedence=300,
                associativity="left",
                to_terms=lambda *term_sets: OrderedSet(
                    functools.reduce(lambda x, y: x * y, term)
                    for term in itertools.product(*term_sets)
                ),
            ),
            Operator(
                "**", arity=2, precedence=500, associativity="right", to_terms=power
            ),
            Operator(
                "^", arity=2, precedence=500, associativity="right", to_terms=power
            ),
        ]

    def resolve(
        self, token: Token, max_prefix_arity: int, context: List[Union[Token, Operator]]
    ) -> Sequence[Operator]:
        if token.token in self.operator_table:
            return super().resolve(token, max_prefix_arity, context)

        symbol = token.token

        # Keep track the number of "+" and "-" characters; if an odd number "-"
        # than "-", else "+"
        while True:
            m = re.search(r"[+\-]{2,}", symbol)
            if not m:
                break
            symbol = (
                symbol[: m.start(0)] + "-"
                if len(m.group(0).replace("+", "")) % 2
                else "+" + symbol[m.end(0) :]
            )

        if symbol in self.operator_table:
            return [self._resolve(token, symbol, max_prefix_arity, context)]

        return [
            self._resolve(token, sym, max_prefix_arity if i == 0 else 0, context)
            for i, sym in enumerate(symbol)
        ]
