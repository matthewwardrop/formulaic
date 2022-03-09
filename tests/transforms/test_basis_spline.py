import re

import numpy
import pytest

from formulaic.transforms.basis_spline import basis_spline
from formulaic import model_matrix
from formulaic.errors import FactorEvaluationError


class TestBasisSpline:
    @pytest.fixture(scope="session")
    def data(self):
        return numpy.linspace(0, 1, 21)

    def test_basic(self, data):
        V = basis_spline(data)

        assert len([k for k in V if isinstance(k, int)]) == 3

        # Comparison data copied from R output of:
        # > library(splines)
        # > data = seq(from=0, to=1, by=0.05)
        # > bs(data)

        assert numpy.allclose(
            V[1],
            [
                0.000000,
                0.135375,
                0.243000,
                0.325125,
                0.384000,
                0.421875,
                0.441000,
                0.443625,
                0.432000,
                0.408375,
                0.375000,
                0.334125,
                0.288000,
                0.238875,
                0.189000,
                0.140625,
                0.096000,
                0.057375,
                0.027000,
                0.007125,
                0.000000,
            ],
        )

        assert numpy.allclose(
            V[2],
            [
                0.000000,
                0.007125,
                0.027000,
                0.057375,
                0.096000,
                0.140625,
                0.189000,
                0.238875,
                0.288000,
                0.334125,
                0.375000,
                0.408375,
                0.432000,
                0.443625,
                0.441000,
                0.421875,
                0.384000,
                0.325125,
                0.243000,
                0.135375,
                0.000000,
            ],
        )

        assert numpy.allclose(
            V[3],
            [
                0.000000,
                0.000125,
                0.001000,
                0.003375,
                0.008000,
                0.015625,
                0.027000,
                0.042875,
                0.064000,
                0.091125,
                0.125000,
                0.166375,
                0.216000,
                0.274625,
                0.343000,
                0.421875,
                0.512000,
                0.614125,
                0.729000,
                0.857375,
                1.000000,
            ],
        )

    def test_degree(self, data):
        V = basis_spline(data, degree=1)

        assert len([k for k in V if isinstance(k, int)]) == 1

        # Comparison data copied from R output of:
        # > library(splines)
        # > data = seq(from=0, to=1, by=0.05)
        # > bs(data, degree=1)

        assert numpy.allclose(
            V[1],
            [
                0.00,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ],
        )

    def test_include_intercept(self, data):
        V = basis_spline(data, degree=1, include_intercept=True)

        assert len([k for k in V if isinstance(k, int)]) == 2

        # Comparison data copied from R output of:
        # > library(splines)
        # > data = seq(from=0, to=1, by=0.05)
        # > bs(data, degree=1, intercept=TRUE)

        assert numpy.allclose(
            V[0],
            [
                1.00,
                0.95,
                0.90,
                0.85,
                0.80,
                0.75,
                0.70,
                0.65,
                0.60,
                0.55,
                0.50,
                0.45,
                0.40,
                0.35,
                0.30,
                0.25,
                0.20,
                0.15,
                0.10,
                0.05,
                0.00,
            ],
        )

        assert numpy.allclose(
            V[1],
            [
                0.00,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ],
        )

    def test_df(self, data):
        V = basis_spline(data, df=5)

        assert len([k for k in V if isinstance(k, int)]) == 5

        # Comparison data copied from R output of:
        # > library(splines)
        # > data = seq(from=0, to=1, by=0.05)
        # > bs(data, df=5)

        assert numpy.allclose(
            V[1],
            [
                0.00000000,
                0.35465625,
                0.54225000,
                0.59821875,
                0.55800000,
                0.45703125,
                0.33075000,
                0.21434375,
                0.12800000,
                0.06865625,
                0.03125000,
                0.01071875,
                0.00200000,
                0.00003125,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ],
        )

        assert numpy.allclose(
            V[2],
            [
                0.00000000,
                0.03065625,
                0.11025000,
                0.22021875,
                0.34200000,
                0.45703125,
                0.54675000,
                0.59278125,
                0.58800000,
                0.54271875,
                0.46875000,
                0.37790625,
                0.28200000,
                0.19284375,
                0.12150000,
                0.07031250,
                0.03600000,
                0.01518750,
                0.00450000,
                0.00056250,
                0.00000000,
            ],
        )

        assert numpy.allclose(
            V[3],
            [
                0.00000000,
                0.00056250,
                0.00450000,
                0.01518750,
                0.03600000,
                0.07031250,
                0.12150000,
                0.19284375,
                0.28200000,
                0.37790625,
                0.46875000,
                0.54271875,
                0.58800000,
                0.59278125,
                0.54675000,
                0.45703125,
                0.34200000,
                0.22021875,
                0.11025000,
                0.03065625,
                0.00000000,
            ],
        )

        assert numpy.allclose(
            V[4],
            [
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00003125,
                0.00200000,
                0.01071875,
                0.03125000,
                0.06865625,
                0.12800000,
                0.21434375,
                0.33075000,
                0.45703125,
                0.55800000,
                0.59821875,
                0.54225000,
                0.35465625,
                0.00000000,
            ],
        )

        assert numpy.allclose(
            V[5],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.001000,
                0.015625,
                0.064000,
                0.166375,
                0.343000,
                0.614125,
                1.000000,
            ],
        )

    def test_extrapolation(self, data):
        # Comparison data based on R output of:
        # > library(splines)
        # > data = seq(from=0, to=1, by=0.05)
        # > bs(data, Boundary.knots=c(0.25, 0.75))

        with pytest.raises(
            ValueError,
            match="Some field values extend beyond upper and/or lower bounds",
        ):
            basis_spline(data, lower_bound=0.25, upper_bound=0.75)

        V = basis_spline(data, lower_bound=0.25, upper_bound=0.75, extrapolation="clip")
        assert numpy.allclose(
            V[3],
            [
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.001,
                0.008,
                0.027,
                0.064,
                0.125,
                0.216,
                0.343,
                0.512,
                0.729,
                1.000,
                1.000,
                1.000,
                1.000,
                1.000,
                1.000,
            ],
        )

        V2 = basis_spline(data, lower_bound=0.25, upper_bound=0.75, extrapolation="na")
        assert numpy.allclose(
            V2[3],
            [
                numpy.nan,
                numpy.nan,
                numpy.nan,
                numpy.nan,
                numpy.nan,
                0.000,
                0.001,
                0.008,
                0.027,
                0.064,
                0.125,
                0.216,
                0.343,
                0.512,
                0.729,
                1.000,
                numpy.nan,
                numpy.nan,
                numpy.nan,
                numpy.nan,
                numpy.nan,
            ],
            equal_nan=True,
        )

        V3 = basis_spline(
            data, lower_bound=0.25, upper_bound=0.75, extrapolation="zero"
        )
        assert numpy.allclose(
            V3[3],
            [
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.001,
                0.008,
                0.027,
                0.064,
                0.125,
                0.216,
                0.343,
                0.512,
                0.729,
                1.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
            ],
            equal_nan=True,
        )

        V4 = basis_spline(
            data, lower_bound=0.25, upper_bound=0.75, extrapolation="extend"
        )
        assert numpy.allclose(
            V4[3],
            [
                -0.125,
                -0.064,
                -0.027,
                -0.008,
                -0.001,
                0.000,
                0.001,
                0.008,
                0.027,
                0.064,
                0.125,
                0.216,
                0.343,
                0.512,
                0.729,
                1.000,
                1.331,
                1.728,
                2.197,
                2.744,
                3.375,
            ],
        )

    def test_invalid_df(self, data):
        with pytest.raises(
            ValueError, match="You cannot specify both `df` and `knots`."
        ):
            basis_spline(data, df=2, knots=[])

        with pytest.raises(
            ValueError, match="Invalid value for `df`. `df` must be greater than 3"
        ):
            basis_spline(data, df=2)

    def test_statefulness(self, data):
        state = {}
        basis_spline(data, _state=state)
        assert state == {
            "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        }

        # Test retention of previous upper and lower bounds by passing in out of
        # bounds data.
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Some field values extend beyond upper and/or lower bounds, which can result in ill-conditioned bases. "
                "Pass a value for `extrapolation` to control how extrapolation should be performed."
            ),
        ):
            basis_spline([-2, 2], extrapolation="raise", _state=state)
