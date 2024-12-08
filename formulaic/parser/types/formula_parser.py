from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import (
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    overload,
)

from typing_extensions import Literal

from formulaic.parser.types.ordered_set import OrderedSet
from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.structured import Structured

from .ast_node import ASTNode
from .operator_resolver import OperatorResolver
from .term import Term
from .token import Token


@dataclass
class FormulaParser:
    """
    The base formula parser API.

    The role of subclasses of this class is to transform a string representation
    of a formula into a (structured) sequence of `Term` instances that can be
    evaluated by materializers and ultimately rendered into model matrices.

    This class can be subclassed to customize this behavior. The three phases of
    formula parsing are split out into separate methods to make this easier.
    They are:
        - get_tokens: Which returns an iterable of `Token` instances. By default
            this uses `tokenize()` and handles the addition/removal of the
            intercept.
        - get_ast: Which converts the iterable of `Token`s into an abstract
            syntax tree. By default this uses `tokens_to_ast()` and the nominated
            `OperatorResolver` instance.
        - get_terms: Which evaluates the abstract syntax tree and returns an
            iterable of `Term`s.
    Only the `get_terms()` method is essential from an API perspective.
    """

    class Target(IntEnum):
        FORMULA = 0
        TOKENS = 1
        AST = 2
        TERMS = 3

    operator_resolver: OperatorResolver
    context: Optional[Mapping[str, Any]] = None

    @overload
    def parse(
        self,
        formula: str,
        *,
        target: Literal[FormulaParser.Target.FORMULA, "formula", 0],
        context: Optional[Mapping[str, Any]] = None,
    ) -> str: ...

    @overload
    def parse(
        self,
        formula: str,
        *,
        target: Literal[FormulaParser.Target.TOKENS, "tokens", 1],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Iterable[Token]: ...

    @overload
    def parse(
        self,
        formula: str,
        *,
        target: Literal[FormulaParser.Target.AST, "ast", 2],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Union[None, Token, ASTNode]: ...

    @overload
    def parse(
        self,
        formula: str,
        *,
        target: Literal[FormulaParser.Target.TERMS, "terms", 3],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Structured[OrderedSet[Term]]: ...

    def parse(
        self,
        formula: str,
        *,
        target: Union[Target, str, int] = Target.TERMS,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Union[
        str, Iterable[Token], Union[None, Token, ASTNode], Structured[OrderedSet[Term]]
    ]:
        """
        Parse the nominated `formula` string to the nominated `target`.

        Args:
            formula: The formula string to be parsed.
            context: An optional context which may be used during the evaluation
                of operators.
        """
        if isinstance(target, int):
            target = self.Target(target)
        elif isinstance(target, str):
            target = self.Target[target.upper()]

        out: Union[
            str,
            Iterable[Token],
            Union[None, Token, ASTNode],
            Structured[OrderedSet[Term]],
        ] = formula
        context = LayeredMapping(context or {}, self.context)
        if target >= self.Target.TOKENS:
            out = tokens = self.get_tokens_from_formula(formula, context=context)
        if target >= self.Target.AST:
            out = ast = self.get_ast_from_tokens(tokens, context=context)
        if target >= self.Target.TERMS:
            out = self.get_terms_from_ast(ast, context=context)
        return out

    def get_tokens_from_formula(
        self, formula: str, *, context: MutableMapping[str, Any]
    ) -> Iterable[Token]:
        """
        Return an iterable of `Token` instances for the nominated `formula`
        string.

        Args:
            formula: The formula string to be tokenized.
        """
        from ..algos.sanitize_tokens import sanitize_tokens
        from ..algos.tokenize import tokenize

        return sanitize_tokens(tokenize(formula))

    def get_ast_from_tokens(
        self, tokens: Iterable[Token], *, context: MutableMapping[str, Any]
    ) -> Union[None, Token, ASTNode]:
        """
        Assemble an abstract syntax tree for the nominated `tokens`.

        Args:
            formula: The formula for which an AST should be generated.
        """
        from ..algos.tokens_to_ast import tokens_to_ast

        return tokens_to_ast(
            tokens,
            operator_resolver=self.operator_resolver,
        )

    def get_terms_from_ast(
        self,
        ast: Union[None, Token, ASTNode],
        *,
        context: MutableMapping[str, Any],
    ) -> Structured[OrderedSet[Term]]:
        """
        Assemble the structured `Term` instances for the nominated AST. A
        `Structured` instance will always be returned even if the structure is
        trivial.

        Args:
            formula: The formula for which an AST should be generated.
            context: An optional context which may be used during the evaluation
                of operators.
        """
        if ast is None:
            return Structured([])

        terms: Union[
            OrderedSet[Term], Tuple[OrderedSet[Term]], Structured[OrderedSet[Term]]
        ] = ast.to_terms(context=context)
        if not isinstance(terms, Structured):
            terms = Structured[OrderedSet[Term]](terms)

        return terms

    # Convenience methods for common use-cases.

    def get_tokens(
        self, formula: str, *, context: Optional[Mapping[str, Any]] = None
    ) -> Iterable[Token]:
        """
        Parse the nominated `formula` string and return the resulting tokens.

        Args:
            formula: The formula string to be parsed.
            context: An optional context which may be used during the evaluation
                of operators.
        """
        return self.parse(formula, target=self.Target.TOKENS, context=context)

    def get_ast(
        self, formula: str, *, context: Optional[Mapping[str, Any]] = None
    ) -> Union[None, Token, ASTNode]:
        """
        Assemble an abstract syntax tree for the nominated `formula` string.

        Args:
            formula: The formula for which an AST should be generated.
            context: An optional context which may be used during the evaluation
                of operators.
        """
        return self.parse(formula, target=self.Target.AST, context=context)

    def get_terms(
        self, formula: str, *, context: Optional[Mapping[str, Any]] = None
    ) -> Structured[OrderedSet[Term]]:
        """
        Parse the nominated `formula` string and return the resulting terms.

        Args:
            formula: The formula string to be parsed.
            context: An optional context which may be used during the evaluation
                of operators.
        """
        return self.parse(formula, target=self.Target.TERMS, context=context)
