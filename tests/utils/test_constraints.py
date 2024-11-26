import re

import numpy
import pytest

from formulaic.errors import FormulaSyntaxError
from formulaic.utils.constraints import LinearConstraintParser, LinearConstraints


class TestLinearConstraints:
    REF_MATRICES = {
        1: [[1, 1, 1]],
        2: [[1, 1, 1], [1, 0, -1]],
    }
    REF_VALUES = {
        1: [0],
        2: [10, 10],
    }

    @pytest.mark.parametrize(
        "check",
        (
            # (<case>, <constraints>)
            (
                1,
                LinearConstraints.from_spec(
                    "a + b + c = 0", variable_names=["a", "b", "c"]
                ),
            ),
            (
                1,
                LinearConstraints.from_spec(
                    {"a + b + c": 0}, variable_names=["a", "b", "c"]
                ),
            ),
            (
                1,
                LinearConstraints.from_spec(
                    {"a + b + c = 10": -10}, variable_names=["a", "b", "c"]
                ),
            ),
            (
                1,
                LinearConstraints.from_spec(
                    {"A[T.a] + b + c = 10": -10}, variable_names=["A[T.a]", "b", "c"]
                ),
            ),
            (1, LinearConstraints.from_spec([1, 1, 1], variable_names=["a", "b", "c"])),
            (
                1,
                LinearConstraints.from_spec(
                    ([1, 1, 1], 0), variable_names=["a", "b", "c"]
                ),
            ),
            (
                1,
                LinearConstraints.from_spec(
                    ([[1, 1, 1]], 0), variable_names=["a", "b", "c"]
                ),
            ),
            (
                1,
                LinearConstraints.from_spec(
                    ([1, 1, 1], [0]), variable_names=["a", "b", "c"]
                ),
            ),
            (
                1,
                LinearConstraints.from_spec(
                    ([[1, 1, 1]], [0]), variable_names=["a", "b", "c"]
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    "a + b + c - 10, a - c = 10", variable_names=["a", "b", "c"]
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    ["a + b + c - 10", "a - c = 10"], variable_names=["a", "b", "c"]
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    {"a + b + c": 10, "a - c": 10}, variable_names=["a", "b", "c"]
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    {"a + b + c = 5": 5, "a - c - 5": 5}, variable_names=["a", "b", "c"]
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    {"A[T.a] + b + c = 5": 5, "A[T.a] - c - 5": 5},
                    variable_names=["A[T.a]", "b", "c"],
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    ([[1, 1, 1], [1, 0, -1]], 10), variable_names=["a", "b", "c"]
                ),
            ),
            (
                2,
                LinearConstraints.from_spec(
                    ([[1, 1, 1], [1, 0, -1]], [10, 10]), variable_names=["a", "b", "c"]
                ),
            ),
        ),
    )
    def test_from_spec(self, check):
        case, constraints = check
        assert numpy.allclose(constraints.constraint_matrix, self.REF_MATRICES[case])
        assert numpy.allclose(constraints.constraint_values, self.REF_VALUES[case])

    def test_from_spec_passthrough(self):
        constraints = LinearConstraints.from_spec(
            "a + b + c = 0", variable_names=["a", "b", "c"]
        )
        assert LinearConstraints.from_spec(constraints) is constraints

    def test_invalid(self):
        with pytest.raises(
            ValueError, match=re.escape("`constraint_matrix` must be a 2D array.")
        ):
            LinearConstraints([[[1, 2, 3]]], 0)
        with pytest.raises(
            ValueError, match=re.escape("`constraint_values` must be a 1D array.")
        ):
            LinearConstraints([[1, 2, 3]], [[1, 2, 3]])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Number of rows in constraint matrix does not equal the number of values in the values array."
            ),
        ):
            LinearConstraints([[1, 2, 3]], [1, 2])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Number of column names does not match the number of columns in the linear constraint matrix."
            ),
        ):
            LinearConstraints([[1, 2, 3]], [0], variable_names=["a"])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "`variable_names` must be provided when parsing constraints from a formula."
            ),
        ):
            LinearConstraints.from_spec("a + b")

    def test_n_constraints(self):
        assert (
            LinearConstraints.from_spec(
                "a = 0", variable_names=["a", "b", "c"]
            ).n_constraints
            == 1
        )
        assert (
            LinearConstraints.from_spec(
                "a = 0, b = 0", variable_names=["a", "b", "c"]
            ).n_constraints
            == 2
        )

    def test_str(self):
        assert (
            str(LinearConstraints.from_spec("a = b", variable_names=["a", "b", "c"]))
            == "1.0 * a + -1.0 * b = 0"
        )
        assert (
            str(
                LinearConstraints.from_spec(
                    "a = b, a = c", variable_names=["a", "b", "c"]
                )
            )
            == "1.0 * a + -1.0 * b = 0\n1.0 * a + -1.0 * c = 0"
        )

    def test_show(self, capsys):
        LinearConstraints.from_spec(
            "a = b, a = c", variable_names=["a", "b", "c"]
        ).show()
        assert (
            capsys.readouterr().out
            == "1.0 * a + -1.0 * b = 0\n1.0 * a + -1.0 * c = 0\n"
        )

    def test_repr(self):
        assert (
            repr(
                LinearConstraints.from_spec(
                    "a = b, a = c", variable_names=["a", "b", "c"]
                )
            )
            == "<LinearConstraints: 2 constraints>"
        )


class TestLinearConstraintParser:
    COLUMNS = list("abcd")

    TEST_CASES = {
        "a": ([[1, 0, 0, 0]], [0]),
        "a + 3 * (a + b - b) / 3 = a": ([[1, 0, 0, 0]], [0]),
        "a + a": ([[2, 0, 0, 0]], [0]),
        "a = 10": ([[1, 0, 0, 0]], [10]),
        "a + b = 10": ([[1, 1, 0, 0]], [10]),
        "a + b - 10": ([[1, 1, 0, 0]], [10]),
        "a + b - 10 = 0": ([[1, 1, 0, 0]], [10]),
        "a = b": ([[1, -1, 0, 0]], [0]),
        "3 * a + b * 3 = 3": ([[3, 3, 0, 0]], [3]),
        "a / 3 + 10 / 2 * d = 0": ([[1 / 3, 0, 0, 5]], [0]),
        "2 * (a + b) - (c + d) / 2": ([[2, 2, -0.5, -0.5]], [0]),
        "a, b, c, d": (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [0, 0, 0, 0],
        ),
        "a = 1, b = 2, c - 3, d - 4": (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [1, 2, 3, 4],
        ),
        "a + b, c + d": ([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 0]),
        "a + b, c + d = 10": ([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 10]),
        "a + b = 5, c + d = 10": ([[1, 1, 0, 0], [0, 0, 1, 1]], [5, 10]),
    }

    @pytest.mark.parametrize("spec,expected", TEST_CASES.items())
    def test_matrix_output(self, spec, expected):
        matrix, values = LinearConstraintParser(self.COLUMNS).get_matrix(spec)
        assert numpy.allclose(matrix, expected[0])
        assert numpy.allclose(values, expected[1])

    def test_empty(self):
        matrix, values = LinearConstraintParser(self.COLUMNS).get_matrix("")
        assert matrix.shape == (0, 4)
        assert values.shape == (0,)

    def test_string_literals(self):
        with pytest.raises(
            FormulaSyntaxError,
            match=re.escape(
                "Only numeric literal values are permitted in constraint formulae."
            ),
        ):
            LinearConstraintParser(self.COLUMNS).get_matrix('"a" * a')

    def test_invalid_cases(self):
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Only one non-scalar factor can be involved in a linear constraint multiplication."
            ),
        ):
            LinearConstraintParser(self.COLUMNS).get_matrix("a * b")
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "The right-hand operand must be a scalar in linear constraint division operations."
            ),
        ):
            LinearConstraintParser(self.COLUMNS).get_matrix("a / b")
