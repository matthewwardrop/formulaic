import pytest

from formulaic.parser.types import Term, Token


class TestToken:
    @pytest.fixture
    def token_a(self):
        return Token("a", kind="name")

    @pytest.fixture
    def token_b(self):
        return Token(
            "log(x)", kind="python", source="y ~ log(x)", source_start=4, source_end=9
        )

    @pytest.fixture
    def token_c(self):
        return Token("+", kind="operator")

    def test_update(self, token_a):
        token_a.update("+", 1, kind="python")
        assert token_a.token == "a+"
        assert token_a.kind.value == "python"
        assert token_a.source_start == 1
        assert token_a.source_end == 1
        assert token_a.source_loc == (1, 1)

    def test_equality(self, token_a, token_b, token_c):
        assert token_a == "a"
        assert token_b == "log(x)"
        assert token_c == "+"

        assert token_a == token_a
        assert token_b != token_a
        assert token_c != token_a

        assert token_a != 1

        assert bool(token_a) == True
        assert bool(Token()) == False

    def test_hash(self, token_a, token_b, token_c):
        assert hash(token_a) == hash("a")

    def test_ranking(self, token_a, token_b, token_c):
        assert token_a < token_b
        assert token_a > token_c

        with pytest.raises(TypeError):
            token_a < 1

    def test_to_factor(self, token_a, token_b, token_c):
        f_a = token_a.to_factor()
        assert f_a.expr == token_a.token
        assert f_a.eval_method.value == "lookup"

        f_b = token_b.to_factor()
        assert f_b.expr == token_b.token
        assert f_b.eval_method.value == "python"

        with pytest.raises(KeyError):
            token_c.to_factor()

    def test_to_terms(self, token_a):
        assert token_a.to_terms() == {Term([token_a.to_factor()])}

    def test_flatten(self, token_a):
        assert token_a.flatten(str_args=False) is token_a
        assert token_a.flatten(str_args=True) is "a"

    def test_get_source_context(self, token_a, token_b, token_c):
        assert token_a.get_source_context() is None
        assert token_b.get_source_context() == "y ~ ⧛log(x)⧚"
        assert token_c.get_source_context() is None

        assert (
            token_b.get_source_context(colorize=True) == "y ~ ⧛\x1b[1;31mlog(x)\x1b[0m⧚"
        )

    def test_copy_with_attrs(self, token_a, token_b):
        assert token_a.copy_with_attrs() is not token_a
        assert token_a.copy_with_attrs().token == "a"
        assert token_a.copy_with_attrs(token="changed").token == "changed"
        assert token_b.copy_with_attrs(token="c").source == token_b.source

    def test_split(self, token_a):
        assert len(list(token_a.split("a"))) == 1
        assert next(iter(token_a.split("a"))) is token_a

        assert list(Token("abc").split("b", after=True)) == [Token("ab"), Token("c")]
        assert list(Token("abc").split("b", before=True)) == [Token("a"), Token("bc")]
        assert list(Token("abc").split("b", after=True, before=True)) == [
            Token("a"),
            Token("b"),
            Token("c"),
        ]

    def test_required_variables(self, token_a, token_b):
        assert token_a.required_variables == {"a"}
        assert token_b.required_variables == {"x"}
        assert Token("a.fillna(0)", kind="python").required_variables == {"a"}
        assert Token("malformed((python", kind="python").required_variables == set()
        assert Token("xyz", kind="value").required_variables == set()
