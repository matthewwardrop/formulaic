import json
import os

import numpy
import numpy as np
import pandas
import pandas as pd
import pytest

from formulaic import model_matrix
from formulaic.transforms.cubic_spline import (
    ExtrapolationError,
    _get_all_sorted_knots,
    _map_cyclic,
    cubic_spline,
    cyclic_cubic_spline,
    natural_cubic_spline,
)

TEST_DATA_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "data")
with open(os.path.join(TEST_DATA_DIR, "cubic_spline_r_test_x.json")) as f:
    cubic_spline_test_x = pandas.Series(json.load(f)).to_numpy()
with open(os.path.join(TEST_DATA_DIR, "cublic_spine_r_test_data.json")) as f:
    cubic_spline_test_data = json.load(f)

for test_set in cubic_spline_test_data:
    for key, val in test_set.items():
        if isinstance(val, dict):
            test_set[key] = numpy.array(
                val["data"], order=val["order"], dtype=val["dtype"]
            )


@pytest.fixture(scope="module", params=cubic_spline_test_data)
def test_data(request):
    return request.param


def test_map_cyclic():
    x = numpy.array([1.5, 2.6, 0.1, 4.4, 10.7])
    x_orig = numpy.copy(x)
    expected_mapped_x = numpy.array([3.0, 2.6, 3.1, 2.9, 3.2])
    mapped_x = _map_cyclic(x, 2.1, 3.6)
    numpy.testing.assert_allclose(x, x_orig)
    numpy.testing.assert_allclose(mapped_x, expected_mapped_x)


def test_map_cyclic_errors():
    import pytest

    x = numpy.linspace(0.2, 5.7, 10)
    with pytest.raises(ValueError):
        _map_cyclic(x, 4.5, 3.6)
    with pytest.raises(ValueError):
        _map_cyclic(x, 4.5, 4.5)


def test_get_all_sorted_knots():
    import pytest

    with pytest.raises(ValueError):
        _get_all_sorted_knots(
            numpy.array([]), n_inner_knots=-1, lower_bound=-1, upper_bound=2
        )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(
            numpy.array([]), n_inner_knots=0, lower_bound=10, upper_bound=-2
        )

    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(
            numpy.array([]), n_inner_knots=0, lower_bound=1, upper_bound=5
        ),
        [1, 5],
    )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(
            numpy.array([]), n_inner_knots=0, lower_bound=1, upper_bound=1
        )

    x = numpy.arange(6) * 2
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, lower_bound=0, upper_bound=100, n_inner_knots=-2)
    lower_bound = x.min()
    upper_bound = x.max()
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, lower_bound, upper_bound, 0), [0, 10]
    )
    numpy.testing.assert_array_equal(_get_all_sorted_knots(x, 3, 8, 0), [3, 8])
    numpy.testing.assert_array_equal(_get_all_sorted_knots(x, 1, 9, 2), [1, 4, 6, 9])
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 1, 3, 2)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 1.3, 1.4, 1)
    numpy.testing.assert_array_equal(_get_all_sorted_knots(x, 1, 3, 1), [1, 2, 3])
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 2, 3, 1)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, lower_bound, upper_bound, 1, inner_knots=[2, 3])
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, lower_bound=2, upper_bound=3)
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, lower_bound, upper_bound, inner_knots=[3, 7]),
        [0, 3, 7, 10],
    )
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, 2, upper_bound, inner_knots=[3, 7]), [2, 3, 7, 10]
    )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(
            x, inner_knots=[3, 7], lower_bound=4, upper_bound=upper_bound
        )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(
            x, inner_knots=[3, 7], upper_bound=6, lower_bound=lower_bound
        )


def test_crs_errors():
    # Invalid 'x' shape
    # TODO: Not ready
    with pytest.raises(ValueError):
        natural_cubic_spline(numpy.arange(16).reshape((4, 4)), df=4, _state={})
    with pytest.raises(ValueError):
        natural_cubic_spline(numpy.arange(16).reshape((4, 4)), df=4, _state={})
    # Should provide at least 'df' or 'knots'
    with pytest.raises(ValueError):
        natural_cubic_spline(numpy.arange(50), cyclic=False)
    # Invalid constraints shape
    with pytest.raises(ValueError):
        natural_cubic_spline(
            numpy.arange(50),
            df=4,
            constraints=numpy.arange(27).reshape((3, 3, 3)),
            cyclic=False,
            _state={},
        )
    # Invalid nb of columns in constraints
    # (should have df + 1 = 5, but 6 provided)
    with pytest.raises(ValueError):
        natural_cubic_spline(
            numpy.arange(50), df=4, constraints=numpy.arange(6), _state={}
        )
    # Too small 'df' for natural cubic spline
    with pytest.raises(ValueError):
        natural_cubic_spline(numpy.arange(50), df=1, _state={})
    # Too small 'df' for cyclic cubic spline
    with pytest.raises(ValueError):
        cyclic_cubic_spline(numpy.arange(50), df=0, _state={})
    with pytest.raises(ValueError, match="Constraints must be"):
        cyclic_cubic_spline(
            numpy.linspace(0, 1, 200),
            df=3,
            constraints="unknown",
            _state={},
        )


def test_crs_with_specific_constraint():
    # Hard coded R values for smooth: s(x, bs="cr", k=5)
    # R> knots <- smooth$xp
    knots_r = numpy.array(
        [
            -2216.837820053100585937,
            -50.456909179687500000,
            -0.250000000000000000,
            33.637939453125000000,
            1477.891880035400390625,
        ]
    )
    # R> centering.constraint <- t(qr.X(attr(smooth, "qrc")))
    context = dict(
        lower_bound=knots_r[0],
        knots=knots_r[1:-1],
        upper_bound=knots_r[-1],
        centering_constraint_r=numpy.array(
            [
                [
                    0.064910676323168478574,
                    1.4519875239407085132,
                    -2.1947446912471946234,
                    1.6129783104357671153,
                    0.064868180547550072235,
                ]
            ]
        ),
    )
    # values for which we want a prediction
    eval_data = pd.DataFrame({"x": numpy.array([-3000.0, -200.0, 300.0, 2000.0])})
    formula = "cr(x, knots=knots,  lower_bound=lower_bound, upper_bound=upper_bound, constraints=centering_constraint_r)"
    result = model_matrix(formula, data=eval_data, context=context)
    train_data = pd.DataFrame({"x": (-1.5) ** numpy.arange(20)})
    intermediate_result = model_matrix(
        "cr(x, df=4, constraints='center')", data=train_data
    )
    result_alt = model_matrix(intermediate_result.model_spec, data=eval_data)
    numpy.testing.assert_allclose(result, result_alt, rtol=1e-12, atol=1e-12)


def test_crs_compat_with_r(test_data):
    # Translate the R output into Python calling conventions
    adjust_df = 0
    if test_data["spline_type"] == "cr" or test_data["spline_type"] == "cs":
        cyclic = False
    else:  #  test_data["spline_type"] == "cc":
        assert test_data["spline_type"] == "cc"
        cyclic = True
        adjust_df += 1

    # Defaults
    constraints = None
    if test_data["absorb_cons"]:
        constraints = "center"
        adjust_df += 1

    df = test_data["nb_knots"] - adjust_df
    knots = test_data["knots"] if test_data["knots"] is not None else None
    lower_bound = (
        test_data["lower_bound"] if test_data["lower_bound"] is not None else None
    )
    upper_bound = (
        test_data["upper_bound"] if test_data["upper_bound"] is not None else None
    )
    if knots is not None:
        # df is not needed when knots are provided
        df = None
    expected_output = test_data["output"]
    state = {}
    out = cubic_spline(
        cubic_spline_test_x,
        df=df,
        knots=knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        constraints=constraints,
        cyclic=cyclic,
        _state=state,
    )
    out_stateful = cubic_spline(
        cubic_spline_test_x,
        df=df,
        knots=knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        constraints=constraints,
        cyclic=cyclic,
        _state=state,
    )

    out_arr = numpy.column_stack(list(out.values()))
    numpy.testing.assert_allclose(out_arr, expected_output, atol=1e-10)
    out_arr_stateful = numpy.column_stack(list(out_stateful.values()))
    numpy.testing.assert_allclose(out_arr, out_arr_stateful, atol=1e-10)

    if cyclic:
        func = cyclic_cubic_spline
    else:
        func = natural_cubic_spline
    out_stateful = func(
        cubic_spline_test_x,
        df=df,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        constraints=constraints,
        _state=state,
    )
    out_stateful_arr = numpy.column_stack(list(out_stateful.values()))
    numpy.testing.assert_allclose(out_stateful_arr, out_arr, atol=1e-10)


def test_statefulness():
    state = {}
    data = numpy.linspace(0.1, 0.9, 50)
    cyclic_cubic_spline(data, df=4, extrapolation="raise", _state=state)
    assert "knots" in state
    knots = state["knots"]
    del state["knots"]
    assert state == {
        "lower_bound": 0.1,
        "upper_bound": 0.9,
        "constraints": None,
        "cyclic": True,
    }
    # Test separately to avoid exact float comparison
    numpy.testing.assert_allclose(knots, [0.1, 0.3, 0.5, 0.7, 0.9])


def test_cubic_spline_edges():
    data = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    data2d = data[:, None]
    state = {}
    res_2d = cyclic_cubic_spline(data2d, df=2, _state={})
    res = cyclic_cubic_spline(data, df=2, _state=state)
    numpy.testing.assert_allclose(res_2d[1], res[1])
    numpy.testing.assert_allclose(res_2d[2], res[2])

    with pytest.raises(ValueError):
        cubic_spline(data, knots=state["knots"], df=2, lower_bound=0.1, upper_bound=0.9)


def test_alternative_extrapolation():
    data = np.linspace(-10.0, 10.0, 21)

    extrap = cyclic_cubic_spline(
        data, df=2, extrapolation="extend", lower_bound=-5.5, upper_bound=5.5, _state={}
    )

    res = cyclic_cubic_spline(
        data, df=2, extrapolation="clip", lower_bound=-5.5, upper_bound=5.5, _state={}
    )
    data_clipped = numpy.clip(data, -5.5, 5.5)
    direct_res = cyclic_cubic_spline(
        data_clipped, df=2, lower_bound=-5.5, upper_bound=5.5, _state={}
    )
    numpy.testing.assert_allclose(res[1], direct_res[1])
    numpy.testing.assert_allclose(res[2], direct_res[2])
    assert not numpy.allclose(extrap[1], res[1])
    assert not numpy.allclose(extrap[2], res[2])

    res = cyclic_cubic_spline(
        data, df=2, extrapolation="na", lower_bound=-5.0, upper_bound=5.0, _state={}
    )
    data_na = numpy.where((data > -5.5) & (data < 5.5), data, np.nan)
    direct_res = cyclic_cubic_spline(data_na, df=2, _state={})
    numpy.testing.assert_allclose(res[1], direct_res[1])
    numpy.testing.assert_allclose(res[2], direct_res[2])
    assert not numpy.allclose(extrap[1], res[1])
    assert not numpy.allclose(extrap[2], res[2])

    with pytest.raises(ExtrapolationError):
        cyclic_cubic_spline(
            data,
            df=2,
            extrapolation="raise",
            lower_bound=-5.5,
            upper_bound=5.5,
            _state={},
        )

    lower_bound = -5.5
    upper_bound = 5.5
    in_bounds = (data >= lower_bound) & (data <= upper_bound)
    valid_data = data[in_bounds]
    state = {}
    res = cyclic_cubic_spline(
        valid_data,
        df=2,
        extrapolation="zero",
        lower_bound=-5.5,
        upper_bound=5.5,
        _state=state,
    )
    re_res = cyclic_cubic_spline(data, extrapolation="zero", _state=state)
    for i in res:
        numpy.testing.assert_allclose(res[i], re_res[i][in_bounds])
        numpy.testing.assert_allclose(
            re_res[i][~in_bounds], numpy.zeros((~in_bounds).sum())
        )
    res2 = cyclic_cubic_spline(
        data, df=2, extrapolation="zero", lower_bound=-5.5, upper_bound=5.5, _state={}
    )
    for i in res:
        numpy.testing.assert_allclose(res2[i], re_res[i])
