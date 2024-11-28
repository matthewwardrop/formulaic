"""
The code in this module is derived from similar code in the `patsy` package.
The original license of the code is as follows:

    Copyright (C) 2011-2012, Patsy Developers. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

       1. Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

       2. Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Iterable, cast

import numpy
import pandas
from typing_extensions import Literal

from formulaic.materializers.types import FactorValues
from formulaic.transforms.basis_spline import SplineExtrapolation
from formulaic.utils.stateful_transforms import stateful_transform


class ExtrapolationError(ValueError):
    pass


def safe_string_eq(obj: Any, value: str) -> bool:
    if isinstance(obj, str):
        return obj == value
    else:
        return False


def _find_knots_lower_bounds(x: numpy.ndarray, knots: numpy.ndarray) -> numpy.ndarray:
    """Finds knots lower bounds for given values.

    Returns an array of indices ``I`` such that
    ``0 <= I[i] <= knots.size - 2`` for all ``i``
    and
    ``knots[I[i]] < x[i] <= knots[I[i] + 1]`` if
    ``numpy.min(knots) < x[i] <= numpy.max(knots)``,
    ``I[i] = 0`` if ``x[i] <= numpy.min(knots)``
    ``I[i] = knots.size - 2`` if ``numpy.max(knots) < x[i]``

    :param x: The 1-d array values whose knots lower bounds are to be found.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: An array of knots lower bounds indices.
    """
    lb = numpy.searchsorted(knots, x) - 1

    # I[i] = 0 for x[i] <= numpy.min(knots)
    lb[lb == -1] = 0

    # I[i] = knots.size - 2 for x[i] > numpy.max(knots)
    lb[lb == knots.size - 1] = knots.size - 2

    return lb


def _compute_base_functions(
    x: numpy.ndarray, knots: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Computes base functions used for building cubic splines basis.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 146
      and for the special treatment of ``x`` values outside ``knots`` range
      see 'mgcv' source code, file 'mgcv.c', function 'crspl()', l.249

    :param x: The 1-d array values for which base functions should be computed.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: 4 arrays corresponding to the 4 base functions ajm, ajp, cjm, cjp
     + the 1-d array of knots lower bounds indices corresponding to
     the given ``x`` values.
    """
    j = _find_knots_lower_bounds(x, knots)

    h = knots[1:] - knots[:-1]
    hj = h[j]
    xj1_x = cast(numpy.ndarray, knots[j + 1] - x)
    x_xj = cast(numpy.ndarray, x - knots[j])

    ajm = xj1_x / hj
    ajp = x_xj / hj

    cjm_3 = xj1_x * xj1_x * xj1_x / (6.0 * hj)
    cjm_3[x > numpy.max(knots)] = 0.0
    cjm_1 = hj * xj1_x / 6.0
    cjm = cjm_3 - cjm_1

    cjp_3 = x_xj * x_xj * x_xj / (6.0 * hj)
    cjp_3[x < numpy.min(knots)] = 0.0
    cjp_1 = hj * x_xj / 6.0
    cjp = cjp_3 - cjp_1

    return ajm, ajp, cjm, cjp, j


def _get_cyclic_f(knots: numpy.ndarray) -> numpy.ndarray:
    """Returns mapping of cyclic cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 146-147

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping cyclic cubic spline values at
     knots to second derivatives.
    """
    h = knots[1:] - knots[:-1]
    n = knots.size - 1
    b = numpy.zeros((n, n))
    d = numpy.zeros((n, n))

    b[0, 0] = (h[n - 1] + h[0]) / 3.0
    b[0, n - 1] = h[n - 1] / 6.0
    b[n - 1, 0] = h[n - 1] / 6.0

    d[0, 0] = -1.0 / h[0] - 1.0 / h[n - 1]
    d[0, n - 1] = 1.0 / h[n - 1]
    d[n - 1, 0] = 1.0 / h[n - 1]

    for i in range(1, n):
        b[i, i] = (h[i - 1] + h[i]) / 3.0
        b[i, i - 1] = h[i - 1] / 6.0
        b[i - 1, i] = h[i - 1] / 6.0

        d[i, i] = -1.0 / h[i - 1] - 1.0 / h[i]
        d[i, i - 1] = 1.0 / h[i - 1]
        d[i - 1, i] = 1.0 / h[i - 1]

    return numpy.linalg.solve(b, d)


def _get_natural_f(knots: numpy.ndarray) -> numpy.ndarray:
    """Returns mapping of natural cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 145-146

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping natural cubic spline values at
     knots to second derivatives.
    """
    from scipy import linalg

    h = knots[1:] - knots[:-1]
    diag = (h[:-1] + h[1:]) / 3.0
    ul_diag = h[1:-1] / 6.0
    banded_b = numpy.array([numpy.r_[0.0, ul_diag], diag, numpy.r_[ul_diag, 0.0]])
    d = numpy.zeros((knots.size - 2, knots.size))
    for i in range(knots.size - 2):
        d[i, i] = 1.0 / h[i]
        d[i, i + 2] = 1.0 / h[i + 1]
        d[i, i + 1] = -d[i, i] - d[i, i + 2]

    fm = linalg.solve_banded((1, 1), banded_b, d)

    return numpy.vstack([numpy.zeros(knots.size), fm, numpy.zeros(knots.size)])


def _map_cyclic(x: numpy.ndarray, lbound: float, ubound: float) -> numpy.ndarray:
    """Maps values into the interval [lbound, ubound] in a cyclic fashion.

    :param x: The 1-d array values to be mapped.
    :param lbound: The lower bound of the interval.
    :param ubound: The upper bound of the interval.
    :return: A new 1-d array containing mapped x values.

    :raise ValueError: if lbound >= ubound.
    """
    if lbound >= ubound:
        raise ValueError(
            f"Invalid argument: lbound ({lbound}) should be "
            f"less than ubound ({ubound})."
        )

    x = numpy.copy(x)
    x[x > ubound] = lbound + (x[x > ubound] - ubound) % (ubound - lbound)
    x[x < lbound] = ubound - (lbound - x[x < lbound]) % (ubound - lbound)

    return x


def _get_free_cubic_spline_matrix(
    x: numpy.ndarray, knots: numpy.ndarray, cyclic: bool = False
) -> numpy.ndarray:
    """Builds an unconstrained cubic regression spline design matrix.

    Returns design matrix with dimensions ``len(x) x n``
    for a cubic regression spline smoother
    where
     - ``n = len(knots)`` for natural CRS
     - ``n = len(knots) - 1`` for cyclic CRS

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 145

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    n = knots.size
    if cyclic:
        x = _map_cyclic(x, min(knots), max(knots))
        n -= 1

    ajm, ajp, cjm, cjp, j = _compute_base_functions(x, knots)

    j1 = j + 1
    if cyclic:
        j1[j1 == n] = 0

    i = numpy.identity(n)

    if cyclic:
        f = _get_cyclic_f(knots)
    else:
        f = _get_natural_f(knots)

    mat_transposed = (
        ajm * i[j, :].T + ajp * i[j1, :].T + cjm * f[j, :].T + cjp * f[j1, :].T
    )

    return mat_transposed.T


def _get_all_sorted_knots(
    x: numpy.ndarray,
    lower_bound: float,
    upper_bound: float,
    n_inner_knots: int | None = None,
    inner_knots: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Gets all knots locations with lower and upper exterior knots included.

    If needed, inner knots are computed as equally spaced quantiles of the
    inumpy.t data falling between given lower and upper bounds.

    :param x: The 1-d array data values.
    :param lower_bound: The lower exterior knot location.
    :param upper_bound: The upper exterior knot location.
    :param n_inner_knots: Number of inner knots to compute.
    :param inner_knots: Provided inner knots if any.
    :return: The array of ``n_inner_knots + 2`` distinct knots.

    :raise ValueError: for various invalid parameters sets or if unable to
     compute ``n_inner_knots + 2`` distinct knots.
    """
    if upper_bound < lower_bound:
        raise ValueError(f"lower_bound > upper_bound ({lower_bound} > {upper_bound})")

    if inner_knots is None and n_inner_knots is not None:
        if n_inner_knots < 0:
            raise ValueError(
                f"Invalid requested number of inner knots: {n_inner_knots}"
            )

        x = x[(lower_bound <= x) & (x <= upper_bound)]
        x = numpy.unique(x)

        if x.size != 0:
            inner_knots_q = numpy.linspace(0, 100, n_inner_knots + 2)[1:-1]
            # .tolist() is necessary to work around a bug in numpy 1.8
            inner_knots = numpy.asarray(numpy.percentile(x, inner_knots_q.tolist()))
        elif n_inner_knots == 0:
            inner_knots = numpy.array([])
        else:
            raise ValueError(
                f"No data values between lower_bound ({lower_bound}) and "
                f"upper_bound ({upper_bound}): cannot compute requested "
                f"{n_inner_knots} inner knot(s)."
            )
    elif inner_knots is not None:
        inner_knots = cast(numpy.ndarray, numpy.unique(inner_knots))
        if n_inner_knots is not None and n_inner_knots != inner_knots.size:
            raise ValueError(
                f"Needed number of inner knots={n_inner_knots} does not match "
                f"provided number of inner knots={inner_knots.size}."
            )
        n_inner_knots = inner_knots.size
        if numpy.any(inner_knots < lower_bound):
            raise ValueError(
                "Some knot values (%s) fall below lower bound "
                "(%r)." % (inner_knots[inner_knots < lower_bound], lower_bound)
            )
        if numpy.any(inner_knots > upper_bound):
            raise ValueError(
                "Some knot values (%s) fall above upper bound "
                "(%r)." % (inner_knots[inner_knots > upper_bound], upper_bound)
            )
    else:
        raise ValueError("Must specify either 'n_inner_knots' or 'inner_knots'.")

    all_knots: numpy.ndarray = numpy.concatenate(
        ([lower_bound, upper_bound], inner_knots)
    )
    all_knots = numpy.unique(all_knots)
    if all_knots.size != n_inner_knots + 2:
        raise ValueError(
            "Unable to compute n_inner_knots(=%r) + 2 distinct "
            "knots: %r data value(s) found between "
            "lower_bound(=%r) and upper_bound(=%r)."
            % (n_inner_knots, x.size, lower_bound, upper_bound)
        )

    return all_knots


def _get_centering_constraint_from_matrix(matrix: numpy.ndarray) -> numpy.ndarray:
    """Computes the centering constraint from the given design matrix.

    We want to ensure that if ``b`` is the array of parameters, our
    model is centered, ie ``numpy.mean(numpy.dot(matrix, b))`` is zero.
    We can rewrite this as ``numpy.dot(c, b)`` being zero with ``c`` a 1-row
    constraint matrix containing the mean of each column of ``matrix``.

    :param matrix: The 2-d array design matrix.
    :return: A 2-d array (1 x ncols(matrix)) defining the
     centering constraint.
    """
    return matrix.mean(axis=0).reshape((1, matrix.shape[1]))


def _absorb_constraints(
    matrix: numpy.ndarray, constraints: numpy.ndarray
) -> numpy.ndarray:
    """Absorb model parameters constraints into the design matrix.

    :param matrix: The (2-d array) initial design matrix.
    :param constraints: The 2-d array defining initial model parameters
     (``betas``) constraints (``numpy.dot(constraints, betas) = 0``).
    :return: The new design matrix with absorbed parameters constraints.

    :raise ImportError: if scipy is not found, used for ``scipy.linalg.qr()``
      which is cleaner than numpy's version requiring a call like
      ``qr(..., mode='complete')`` to get a full QR decomposition.
    """
    m = constraints.shape[0]
    q, r = numpy.linalg.qr(numpy.transpose(constraints), mode="complete")

    return numpy.dot(matrix, q[:, m:])


def _get_cubic_spline_matrix(
    x: numpy.ndarray,
    knots: numpy.ndarray,
    constraints: numpy.ndarray | None = None,
    cyclic: bool = False,
) -> numpy.ndarray:
    """Builds a cubic regression spline design matrix.

    Returns design matrix with dimensions len(x) x n
    where:
     - ``n = len(knots) - nrows(constraints)`` for natural CRS
     - ``n = len(knots) - nrows(constraints) - 1`` for cyclic CRS
    for a cubic regression spline smoother

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param constraints: The 2-d array defining model parameters (``betas``)
     constraints (``numpy.dot(constraints, betas) = 0``).
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    mat = _get_free_cubic_spline_matrix(x, knots, cyclic)
    if constraints is not None:
        mat = _absorb_constraints(mat, constraints)

    return mat


def cubic_spline(  # pylint: disable=dangerous-default-value  # always replaced by stateful-transform
    x: pandas.Series | numpy.ndarray,
    df: int | None = None,
    knots: Iterable[float] | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    constraints: numpy.ndarray | Literal["center"] | None = None,
    extrapolation: str | SplineExtrapolation = "extend",
    cyclic: bool = True,
    _state: dict = {},
) -> FactorValues[dict]:
    """
    Evaluates the B-Spline basis vectors for given inumpy.ts `x`.

    This is especially useful in the context of allowing non-linear fits to data
    in linear regression. Except for the addition of the `extrapolation`
    parameter, this implementation shares its API with `patsy.splines.bs`, and
    should behave identically to both `patsy.splines.bs` and R's `splines::bs`
    where functionality overlaps.

    Args:
        x: The vector for which the B-Spline basis should be computed.
        df: The number of degrees of freedom to use for this spline. If
            specified, `knots` will be automatically generated such that they
            are `df` - `degree` (minus one if `include_intercept` is True)
            equally spaced quantiles. You cannot specify both `df` and `knots`.
        knots: The internal breakpoints of the B-Spline. If not specified, they
            default to the empty list (unless `df` is specified), in which case
            the ordinary polynomial (Bezier) basis is generated.
        lower_bound: The lower bound for the domain for the B-Spline basis. If
            not specified this is determined from `x`.
        upper_bound: The upper bound for the domain for the B-Spline basis. If
            not specified this is determined from `x`.
        extrapolation: Selects how extrapolation should be performed when values
            in `x` extend beyond the lower and upper bounds. Valid values are:
            - 'raise': Raises a `ValueError` if there are any values in `x`
              outside the B-Spline domain.
            - 'clip': Any values above/below the domain are set to the
              upper/lower bounds.
            - 'na': Any values outside of bounds are set to `numpy.nan`.
            - 'zero': Any values outside of bounds are set to `0`.
            - 'extend': Any values outside of bounds are computed by extending
              the polynomials of the B-Spline (this is the same as the default
              in R).

    Returns:
        A dictionary representing the encoded vectors ready for ingestion
        by materializers (wrapped in a `FactorValues` instance providing
        relevant metadata).

    Notes:
        The implementation employed here uses a slightly generalised version of
        the ["Cox-de Boor" algorithm](https://en.wikipedia.org/wiki/B-spline#Definition),
        extended by this author to allow for extrapolations (although this
        author doubts this is terribly novel). We have not used the `splev`
        methods from `scipy` since in benchmarks this implementation outperforms
        them for our use-cases.

        If you would like to learn more about B-Splines, the primer put together
        by Jeffrey Racine is an excellent resource:
        https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf

        As a stateful transform, we only keep track of `knots`, `lower_bound`
        and `upper_bound`, which are sufficient given that all other information
        must be explicitly specified.
    """
    # Prepare and check arguments
    if df is not None and knots is not None:
        raise ValueError("You cannot specify both `df` and `knots`.")

    if "lower_bound" in _state:
        lower_bound = float(_state["lower_bound"])
    else:
        lower_bound = _state["lower_bound"] = (
            float(numpy.min(x)) if lower_bound is None else float(lower_bound)
        )

    if "upper_bound" in _state:
        upper_bound = float(_state["upper_bound"])
    else:
        upper_bound = _state["upper_bound"] = (
            float(numpy.max(x)) if upper_bound is None else float(upper_bound)
        )

    if "constraints" in _state:
        constraints = _state["constraints"]
    else:
        _state["constraints"] = constraints

    if "cyclic" in _state:
        cyclic = _state["cyclic"]
    else:
        _state["cyclic"] = cyclic
    # Check and reformat x
    x = numpy.atleast_1d(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim > 1:
        raise ValueError(
            "Inumpy.t to cubic_spline must be 1-d, or a 2-d column vector."
        )

    if "extrapolation" in _state:
        extrapolation = SplineExtrapolation(_state["extrapolation"])
    else:
        extrapolation = SplineExtrapolation(extrapolation)
        _state["extrapolation"] = extrapolation.value

    # Check extrapolations and adjust x if necessary
    # SplineExtrapolation.EXTEND is the natural default, so no need to do anything
    if extrapolation is SplineExtrapolation.RAISE and numpy.any(
        (x < lower_bound) | (x > upper_bound)
    ):
        raise ExtrapolationError(
            "Some field values extend beyond upper and/or lower bounds, which can "
            "result in ill-conditioned bases. Pass a value for `extrapolation` to "
            "control how extrapolation should be performed."
        )
    elif extrapolation is SplineExtrapolation.CLIP:
        x = numpy.clip(x, lower_bound, upper_bound)
    elif extrapolation in (SplineExtrapolation.NA, SplineExtrapolation.ZERO):
        fill_value = numpy.nan if extrapolation is SplineExtrapolation.NA else 0.0
        x = numpy.where((x >= lower_bound) & (x <= upper_bound), x, fill_value)

    # Prepare knots
    if "knots" not in _state:
        if df is None and knots is None:
            raise ValueError("Must specify either 'df' or 'knots'.")

        n_constraints = 0
        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                # Here we collect only number of constraints,
                # actual centering constraint will be computed after all_knots
                n_constraints = 1
            else:
                constraints_arr = numpy.atleast_2d(constraints)
                if constraints_arr.ndim != 2:
                    raise ValueError("Constraints must be 2-d array or 1-d vector.")
                n_constraints = constraints_arr.shape[0]

        n_inner_knots = None
        if df is not None:
            min_df = 1
            if not cyclic and n_constraints == 0:
                min_df = 2
            if df < min_df:
                raise ValueError(
                    f"'df'={df} must be greater than or equal to {min_df}."
                )
            n_inner_knots = df - 2 + n_constraints
            if cyclic:
                n_inner_knots += 1
        _knots = numpy.array(knots) if knots is not None else None
        all_knots = _get_all_sorted_knots(
            x,
            lower_bound,
            upper_bound,
            n_inner_knots=n_inner_knots,
            inner_knots=_knots,
        )
        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                # Now we can compute centering constraints
                constraints_arr = _get_centering_constraint_from_matrix(
                    _get_free_cubic_spline_matrix(x, all_knots, cyclic=cyclic)
                )

            df_before_constraints = all_knots.size
            if cyclic:
                df_before_constraints -= 1
            if constraints_arr.shape[1] != df_before_constraints:
                raise ValueError(
                    f"Constraints array should have {df_before_constraints} columns "
                    f"but {constraints_arr.shape[1]} found."
                )
            _state["constraints"] = constraints_arr
        _state["knots"] = all_knots.tolist()
    constraints = _state["constraints"]
    knots = numpy.array(_state["knots"])

    # Compute cubic splines
    cs_mat = _get_cubic_spline_matrix(x, knots, constraints, cyclic=cyclic)

    return FactorValues(
        {i + 1: cs_mat[:, i] for i in range(cs_mat.shape[1])},
        kind="numerical",
        spans_intercept=False,
        drop_field=0,
        format="{name}[{field}]",
        encoded=False,
    )


cyclic_cubic_spline = stateful_transform(partial(cubic_spline, cyclic=True))
natural_cubic_spline = stateful_transform(partial(cubic_spline, cyclic=False))
