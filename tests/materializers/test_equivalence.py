from typing import Any

import pandas
import pytest
import scipy.sparse

from formulaic.materializers import FormulaMaterializer

DATASET = pandas.DataFrame(
    {
        "a": [1, 2, None],
        "b": [1, 2, 3],
        "A": ["a", None, "c"],
        "B": ["a", "b", None],
        "D": ["a", "a", "a"],
    }
)

TEST_CASES = {
    # "name": expected_output
    "a": pandas.DataFrame(
        {
            "Intercept": [1, 1],
            "a": [1, 2],
        }
    ),
    "A": pandas.DataFrame(
        {
            "Intercept": [1, 1],
            "A[T.c]": [0, 1],
        }
    ),
    "bs(b)": pandas.DataFrame(
        {
            "Intercept": [1, 1, 1],
            "bs(b)[1]": [0.0, 0.375, 0.0],
            "bs(b)[2]": [0.0, 0.375, 0.0],
            "bs(b)[3]": [0.0, 0.125, 1.0],
        }
    ),
    "C(b)": pandas.DataFrame(
        {
            "Intercept": [1, 1, 1],
            "C(b)[T.2]": [0, 1, 0],
            "C(b)[T.3]": [0, 0, 1],
        }
    ),
    # "cs(b, df=2)": pandas.DataFrame({
    #     "Intercept": [1, 1, 1],
    #     "cs(b)[1]": [0.0, 0.5, 0.0],
    #     "cs(b)[2]": [0.0, 0.0, 0.0],
    #     "cs(b)[3]": [0.0, 0.5, 1.0],
    # }),
    "hashed(b, levels=2)": pandas.DataFrame(
        {
            "Intercept": [1, 1, 1],
            "hashed(b, levels=2)[0]": [0, 1, 0],
            "hashed(b, levels=2)[1]": [1, 0, 1],
        }
    ),
    "I(a)": pandas.DataFrame(
        {
            "Intercept": [1, 1],
            "I(a)": [1, 2],
        }
    ),
    "a + lag(a)": pandas.DataFrame(
        {
            "Intercept": [1],
            "a": [2],
            "lag(a)": [1],
        }
    ),
    "poly(b, degree=2)": pandas.DataFrame(
        {
            "Intercept": [1, 1, 1],
            "poly(b, degree=2)[1]": [-0.707107, 0.0, 0.707107],
            "poly(b, degree=2)[2]": [0.408248, -0.816497, 0.408248],
        }
    ),
    "scale(b)": pandas.DataFrame(
        {
            "Intercept": [1, 1, 1],
            "scale(b)": [-1, 0, 1],
        }
    ),
}


class TestMaterializerEquivalence:
    # Materializer fixtures

    @pytest.fixture(
        autouse=True,
        params=[
            (materializer_name, input, output)
            for materializer_name, materializer in FormulaMaterializer.REGISTERED_NAMES.items()
            for input in ("pandas", "polars")
            for output in materializer.REGISTER_OUTPUTS
            if not (input == "polars" and materializer_name == "pandas")
        ],
        ids=lambda x: f"{x[0]}|{x[1]}->{x[2]}",
    )
    def materializer_case(self, request):
        return request.param

    @pytest.fixture
    def materializer(self, materializer_case):
        materializer_name, _, _ = materializer_case
        return FormulaMaterializer.REGISTERED_NAMES[materializer_name]

    @pytest.fixture
    def input_type(self, materializer_case):
        _, input_type, _ = materializer_case
        return input_type

    @pytest.fixture
    def output_type(self, materializer_case):
        _, _, output_type = materializer_case
        return output_type

    # Data fixtures

    @pytest.fixture
    def df(self, input_type):
        data = DATASET
        if input_type == "pandas":
            return data
        elif input_type == "polars":
            try:
                import polars
            except ImportError:
                pytest.skip("polars is not installed")

            return polars.from_pandas(data)
        raise ValueError(f"Unknown input_type: {input_type}")

    @pytest.mark.parametrize(
        "formula, expected_output", TEST_CASES.items(), ids=list(TEST_CASES)
    )
    def test_equivalence(self, materializer, df, output_type, formula, expected_output):
        if not materializer.SUPPORTS_INPUT(df):
            pytest.skip(
                f"Materializer {materializer} does not support input type {type(df)}"
            )

        result = materializer(df).get_model_matrix(formula, output=output_type)

        assert_equivalent(expected_output, result, output_type)


def assert_equivalent(df1: pandas.DataFrame, df2: Any, output_type: str):
    import narwhals
    import numpy

    try:
        import polars
    except ImportError:
        polars = None

    if isinstance(df2, scipy.sparse.spmatrix):
        assert output_type == "sparse"
        df2 = pandas.DataFrame(df2.toarray(), columns=df2.model_spec.column_names)
    elif isinstance(df2, numpy.ndarray):
        assert output_type == "numpy"
        df2 = pandas.DataFrame(df2, columns=df2.model_spec.column_names)
    elif polars and isinstance(df2, polars.DataFrame):
        assert output_type == "narwhals"
        df2 = df2.to_pandas()
    elif isinstance(df2, narwhals.DataFrame):
        assert output_type == "narwhals"
        df2 = df2.to_pandas()
    else:
        assert output_type in ("pandas", "narwhals")
    pandas.testing.assert_frame_equal(
        df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False
    )
