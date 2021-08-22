import pytest

from formulaic.parser.types import ASTNode, Operator


class TestASTNode:
    @pytest.fixture
    def ast_node(self):
        return ASTNode(
            Operator("+", arity=2, precedence=100, to_terms=lambda *args: args),
            ("a", "b", "c"),
        )

    def test_to_terms(self, ast_node):
        assert ast_node.to_terms() == ("a", "b", "c")

    def test_flatten(self, ast_node):
        assert ast_node.flatten(str_args=True) == ["+", "a", "b", "c"]

    def test_repr(self, ast_node):
        assert repr(ast_node) == "<ASTNode +: ('a', 'b', 'c')>"
