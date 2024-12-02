import pytest

from formulaic.errors import FormulaParsingError, FormulaSyntaxError
from formulaic.parser.types import Operator, OperatorResolver, Token

OPERATOR_PLUS = Operator("+", arity=2, precedence=100, fixity="infix")
OPERATOR_UNARY_MINUS = Operator("-", arity=1, precedence=100, fixity="prefix")
OPERATOR_COLON = Operator(":", arity=1, precedence=100, fixity="postfix")
OPERATOR_COLON_2 = Operator(":", arity=2, precedence=100, fixity="infix")


class DummyOperatorResolver(OperatorResolver):
    @property
    def operators(self):
        return [
            OPERATOR_PLUS,
            OPERATOR_UNARY_MINUS,
            OPERATOR_COLON,
            OPERATOR_COLON_2,
        ]


class TestOperatorResolver:
    @pytest.fixture
    def resolver(self):
        return DummyOperatorResolver()

    def test_resolve(self, resolver):
        assert list(resolver.resolve(Token("+")))[0][1][0] is OPERATOR_PLUS
        assert list(resolver.resolve(Token("-")))[0][1][0] is OPERATOR_UNARY_MINUS
