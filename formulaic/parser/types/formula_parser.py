from dataclasses import dataclass
from typing import Iterable, List

from .ast_node import ASTNode
from .operator_resolver import OperatorResolver
from .structured import Structured
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

    operator_resolver: OperatorResolver

    def get_tokens(self, formula: str) -> Iterable[Token]:
        """
        Return an iterable of `Token` instances for the nominated `formula`
        string.

        Args:
            formula: The formula string to be tokenized.
        """
        from ..algos.tokenize import tokenize

        return tokenize(formula)

    def get_ast(self, formula: str) -> ASTNode:
        """
        Assemble an abstract syntax tree for the nominated `formula` string.

        Args:
            formula: The formula for which an AST should be generated.
        """
        from ..algos.tokens_to_ast import tokens_to_ast

        return tokens_to_ast(
            self.get_tokens(formula),
            operator_resolver=self.operator_resolver,
        )

    def get_terms(self, formula: str, *, sort: bool = True) -> Structured[List[Term]]:
        """
        Assemble the `Term` instances for a formula string. Depending on the
        operators involved, this may be an iterable of `Term` instances, or
        an iterable of iterables of `Term`s, etc.

        Args:
            formula: The formula for which an AST should be generated.
            sort: Whether to sort the terms before returning them.
        """
        ast = self.get_ast(formula)
        if ast is None:
            return Structured([])

        terms = ast.to_terms()
        if not isinstance(terms, Structured):
            terms = Structured(terms)

        if sort:
            terms = terms._map(sorted)

        return terms
