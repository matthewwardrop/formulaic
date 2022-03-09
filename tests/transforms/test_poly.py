import numpy
import pytest

from formulaic.transforms.poly import poly


class TestPoly:
    @pytest.fixture(scope="session")
    def data(self):
        return numpy.linspace(0, 1, 21)

    def test_basic(self, data):
        state = {}
        V = poly(data, _state=state)

        assert V.shape[1] == 1

        # Comparison data copied from R output of:
        # > data = seq(from=0, to=1, by=0.05)
        # > poly(data)
        r_reference = [
            -3.603750e-01,
            -3.243375e-01,
            -2.883000e-01,
            -2.522625e-01,
            -2.162250e-01,
            -1.801875e-01,
            -1.441500e-01,
            -1.081125e-01,
            -7.207500e-02,
            -3.603750e-02,
            -3.000725e-17,
            3.603750e-02,
            7.207500e-02,
            1.081125e-01,
            1.441500e-01,
            1.801875e-01,
            2.162250e-01,
            2.522625e-01,
            2.883000e-01,
            3.243375e-01,
            3.603750e-01,
        ]

        assert numpy.allclose(
            V[:, 0],
            r_reference,
        )

        assert pytest.approx(state["alpha"], {0: 0.5})
        assert pytest.approx(state["norms2"], {0: 21.0, 2: 1.925})

        assert numpy.allclose(
            poly(data, _state=state)[:, 0],
            r_reference,
        )

    def test_degree(self, data):
        state = {}
        V = poly(data, degree=3, _state=state)

        assert V.shape[1] == 3

        # Comparison data copied from R output of:
        # > data = seq(from=0, to=1, by=0.05)
        # > poly(data, 3)
        r_reference = numpy.array(
            [
                [-3.603750e-01, 0.42285541, -4.332979e-01],
                [-3.243375e-01, 0.29599879, -1.733191e-01],
                [-2.883000e-01, 0.18249549, 1.824412e-02],
                [-2.522625e-01, 0.08234553, 1.489937e-01],
                [-2.162250e-01, -0.00445111, 2.265312e-01],
                [-1.801875e-01, -0.07789442, 2.584584e-01],
                [-1.441500e-01, -0.13798440, 2.523770e-01],
                [-1.081125e-01, -0.18472105, 2.158888e-01],
                [-7.207500e-02, -0.21810437, 1.565954e-01],
                [-3.603750e-02, -0.23813436, 8.209854e-02],
                [-3.000725e-17, -0.24481103, -4.395626e-17],
                [3.603750e-02, -0.23813436, -8.209854e-02],
                [7.207500e-02, -0.21810437, -1.565954e-01],
                [1.081125e-01, -0.18472105, -2.158888e-01],
                [1.441500e-01, -0.13798440, -2.523770e-01],
                [1.801875e-01, -0.07789442, -2.584584e-01],
                [2.162250e-01, -0.00445111, -2.265312e-01],
                [2.522625e-01, 0.08234553, -1.489937e-01],
                [2.883000e-01, 0.18249549, -1.824412e-02],
                [3.243375e-01, 0.29599879, 1.733191e-01],
                [3.603750e-01, 0.42285541, 4.332979e-01],
            ]
        )

        assert numpy.allclose(
            V,
            r_reference,
        )

        assert pytest.approx(state["alpha"], {0: 0.5, 1: 0.5, 2: 0.5})
        assert pytest.approx(
            state["norms2"], {1: 0.09166666666666667, 2: 0.07283333333333333}
        )

        assert numpy.allclose(
            poly(data, degree=3, _state=state),
            r_reference,
        )

    def test_reuse_state(self, data):
        state = {}
        V = poly(data, degree=3, _state=state)  # as tested above

        # Reuse state but with different data
        V = poly(data**2, degree=3, _state=state)

        assert V.shape[1] == 3

        # Comparison data copied from R output of:
        # > data = seq(from=0, to=1, by=0.05)
        # > coefs = attr(poly(data, 3), 'coefs')
        # > poly(data^2, 3, coefs=coefs)
        r_reference = numpy.array(
            [
                [-0.36037499, 0.422855413, -0.43329786],
                [-0.35857311, 0.416195441, -0.41855671],
                [-0.35316749, 0.396415822, -0.37546400],
                [-0.34415811, 0.364117458, -0.30735499],
                [-0.33154499, 0.320301848, -0.21959840],
                [-0.31532811, 0.266371091, -0.11931132],
                [-0.29550749, 0.204127887, -0.01496018],
                [-0.27208311, 0.135775535, 0.08415243],
                [-0.24505499, 0.063917934, 0.16851486],
                [-0.21442312, -0.008440417, 0.22914758],
                [-0.18018749, -0.077894418, 0.25845837],
                [-0.14234812, -0.140638372, 0.25121156],
                [-0.10090500, -0.192465980, 0.20561124],
                [-0.05585812, -0.228770342, 0.12449854],
                [-0.00720750, -0.244543962, 0.01666296],
                [0.04504687, -0.234378741, -0.10173235],
                [0.10090500, -0.192465980, -0.20561124],
                [0.16036687, -0.112596382, -0.25933114],
                [0.22343249, 0.011839952, -0.21491574],
                [0.29010186, 0.187853517, -0.01017347],
                [0.36037499, 0.422855413, 0.43329786],
            ]
        )

        assert numpy.allclose(
            V,
            r_reference,
        )

    def test_raw(self, data):
        assert numpy.allclose(
            poly(data, 3, raw=True), numpy.array([data, data**2, data**3]).T
        )
