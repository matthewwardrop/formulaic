import numpy as np
import numpy.testing as npt
import pytest
import pandas as pd

from formulaic.materializers.transforms import basis_splines


def test_expected_output_in_dict():
    state = {}
    output = basis_splines(data=[0, 0, 0], n_knots=1, state=state)
    assert 'bs_transform' in state
    npt.assert_allclose(output[0].values, np.array([0, 0, 0]))


def test_basis_spline_accepts_kwargs():
    state = {}
    input_data = [1, 2, 3]
    basis_splines(data=input_data, n_knots=2, knots=[1.5, 2.5], term=1, restricted=False, state=state)


    npt.assert_allclose(
            state['bs_transform'](input_data),
            np.array([
                [0.0, 0.0],
                [0.5, 0.0],
                [1.5, 0.5]
            ])
        )


def test_basis_spline_keeps_state_and_reuses_it():
    state = {}

    input_data = [1, 5, 10, 15, 20]
    expected_output = np.array([[0.0, 0.0, (10.0 - 5.0)**2 - 0, (15.0 - 5.0)**2 - 0, (20.0 - 5.0)**2 - (20.0 - 16.0)**2]]).T

    basis_splines(data=input_data, n_knots=2, knots=[5, 16], term=2, restricted=True, state=state)
    assert 'bs_transform' in state

    npt.assert_allclose(
            state['bs_transform'](input_data),
            expected_output
        )

    npt.assert_allclose(
            state['bs_transform'](input_data[2:4]),
            expected_output[2:4]
    )


def test_automatic_knots():
    state = {}
    input_data = [1, 5, 10, 15, 20]

    basis_splines(data=input_data, n_knots=2, restricted=False, state=state)

    expected_output = state['bs_transform'](input_data).values
    expected_splines = np.array([[0.0,       0.0],
                                [0.0,       0.0],
                                [10 - 20/3, 0.0],
                                [15 - 20/3, 15 - 40/3],
                                [20 - 20/3, 20 - 40/3],
                                ])
    npt.assert_allclose(expected_output, expected_splines)


def test_error_is_thrown_if_too_many_knots():
    with pytest.raises(ValueError, match="integer between 1 and 7"):
        basis_splines([0, 0, 0], n_knots=np.inf)
