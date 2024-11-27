import json
import os

import numpy
import pandas
import pytest

from formulaic.transforms.cubic_spline import (
    CC,
    CR,
    _get_all_sorted_knots,
    _map_cyclic,
    cc,
    cr,
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
        _get_all_sorted_knots(numpy.array([]), -1)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(numpy.array([]), 0)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(numpy.array([]), 0, lower_bound=1)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(numpy.array([]), 0, upper_bound=5)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(numpy.array([]), 0, lower_bound=3, upper_bound=1)

    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(numpy.array([]), 0, lower_bound=1, upper_bound=5), [1, 5]
    )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(numpy.array([]), 0, lower_bound=1, upper_bound=1)

    x = numpy.arange(6) * 2
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, -2)
    numpy.testing.assert_array_equal(_get_all_sorted_knots(x, 0), [0, 10])
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, 0, lower_bound=3, upper_bound=8), [3, 8]
    )
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, 2, lower_bound=1, upper_bound=9), [1, 4, 6, 9]
    )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 2, lower_bound=1, upper_bound=3)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 1, lower_bound=1.3, upper_bound=1.4)
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, 1, lower_bound=1, upper_bound=3), [1, 2, 3]
    )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 1, lower_bound=2, upper_bound=3)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, 1, inner_knots=[2, 3])
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, lower_bound=2, upper_bound=3)
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, inner_knots=[3, 7]), [0, 3, 7, 10]
    )
    numpy.testing.assert_array_equal(
        _get_all_sorted_knots(x, inner_knots=[3, 7], lower_bound=2), [2, 3, 7, 10]
    )
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, inner_knots=[3, 7], lower_bound=4)
    with pytest.raises(ValueError):
        _get_all_sorted_knots(x, inner_knots=[3, 7], upper_bound=6)


def test_crs_errors():
    import pytest

    # Invalid 'x' shape
    # TODO: Not ready
    with pytest.raises(ValueError):
        CR().transform(numpy.arange(16).reshape((4, 4)), df=4)
    with pytest.raises(ValueError):
        CR().transform(numpy.arange(16).reshape((4, 4)), df=4)
    # Should provide at least 'df' or 'knots'
    with pytest.raises(ValueError):
        _cr = CR()
        _cr.memorize_chunk(numpy.arange(50))
        _cr.memorize_finish()
    # Invalid constraints shape
    with pytest.raises(ValueError):
        _cr = CR()
        _cr.memorize_chunk(
            numpy.arange(50),
            df=4,
            constraints=numpy.arange(27).reshape((3, 3, 3)),
        )
        _cr.memorize_finish()
    # Invalid nb of columns in constraints
    # (should have df + 1 = 5, but 6 provided)
    with pytest.raises(ValueError):
        _cr = CR()
        _cr.memorize_chunk(numpy.arange(50), df=4, constraints=numpy.arange(6))
        _cr.memorize_finish()
    # Too small 'df' for natural cubic spline
    with pytest.raises(ValueError):
        _cr = CR()
        _cr.memorize_chunk(numpy.arange(50), df=1)
        _cr.memorize_finish()
    # Too small 'df' for cyclic cubic spline
    with pytest.raises(ValueError):
        _cr = CR()
        _cr.memorize_chunk(numpy.arange(50), df=0)
        _cr.memorize_finish()


def test_crs_compat(test_data):
    # Translate the R output into Python calling conventions
    adjust_df = 0
    if test_data["spline_type"] == "cr" or test_data["spline_type"] == "cs":
        spline_type = CR
    else:  #  test_data["spline_type"] == "cc":
        spline_type = CC
        adjust_df += 1

    # Defaults
    df = constraints = None
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
    t = spline_type()
    t.memorize_chunk(
        cubic_spline_test_x,
        df=df,
        knots=knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        constraints=constraints,
    )
    t.memorize_finish()
    output = t.transform(
        cubic_spline_test_x,
        df=df,
        knots=knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        constraints=constraints,
    )
    assert output.shape[0] == len(cubic_spline_test_x)
    numpy.testing.assert_allclose(output, expected_output, atol=1e-10)


@pytest.mark.skip(reason="Not ready yet")
def test_crs_with_specific_constraint():
    from patsy.highlevel import build_design_matrices, dmatrix, incr_dbuilder

    x = (-1.5) ** numpy.arange(20)
    # Hard coded R values for smooth: s(x, bs="cr", k=5)
    # R> knots <- smooth$xp
    knots_R = numpy.array(
        [
            -2216.837820053100585937,
            -50.456909179687500000,
            -0.250000000000000000,
            33.637939453125000000,
            1477.891880035400390625,
        ]
    )
    # R> centering.constraint <- t(qr.X(attr(smooth, "qrc")))
    centering_constraint_R = numpy.array(
        [
            [
                0.064910676323168478574,
                1.4519875239407085132,
                -2.1947446912471946234,
                1.6129783104357671153,
                0.064868180547550072235,
            ]
        ]
    )
    # values for which we want a prediction
    new_x = numpy.array([-3000.0, -200.0, 300.0, 2000.0])
    result1 = dmatrix(
        "cr(new_x, knots=knots_R[1:-1], "
        "lower_bound=knots_R[0], upper_bound=knots_R[-1], "
        "constraints=centering_constraint_R)"
    )

    data_chunked = [{"x": x[:10]}, {"x": x[10:]}]
    new_data = {"x": new_x}
    builder = incr_dbuilder(
        "cr(x, df=4, constraints='center')", lambda: iter(data_chunked)
    )
    result2 = build_design_matrices([builder], new_data)[0]

    assert numpy.allclose(result1, result2, rtol=1e-12, atol=0.0)
