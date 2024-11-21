from __future__ import annotations
from typing import Optional, Iterable, Union
import numpy
import pandas
from formulaic.materializers.types import FactorValues
from formulaic.utils.stateful_transforms import stateful_transform


df: Optional[int] = None,
knots: Optional[Iterable[float]] = None,
degree: int = 3,
include_intercept: bool = False,
lower_bound: Optional[float] = None,
upper_bound: Optional[float] = None,
extrapolation: Union[str, SplineExtrapolation] = "raise",
_state: dict = {},



def _find_knots_lower_bounds(x, knots):
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


def _map_cyclic(x, lbound, ubound):
    """Maps values into the interval [lbound, ubound] in a cyclic fashion.

    :param x: The 1-d array values to be mapped.
    :param lbound: The lower bound of the interval.
    :param ubound: The upper bound of the interval.
    :return: A new 1-d array containing mapped x values.

    :raise ValueError: if lbound >= ubound.
    """
    if lbound >= ubound:
        raise ValueError(
            "Invalid argument: lbound (%r) should be "
            "less than ubound (%r)." % (lbound, ubound)
        )

    x = numpy.copy(x)
    x[x > ubound] = lbound + (x[x > ubound] - ubound) % (ubound - lbound)
    x[x < lbound] = ubound - (lbound - x[x < lbound]) % (ubound - lbound)

    return x

def _compute_base_functions(x, knots):
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
    xj1_x = knots[j + 1] - x
    x_xj = x - knots[j]

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


def _get_cyclic_f(knots):
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

def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 145-146

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping natural cubic spline values at
     knots to second derivatives.

    :raise ImportError: if scipy is not found, required for
     ``linalg.solve_banded()``
    """
    try:
        from scipy import linalg
    except ImportError:  # pragma: no cover
        raise ImportError("Cubic spline functionality requires scipy.")

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


def _get_free_crs_dmatrix(x, knots, cyclic=False):
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

    dmt = ajm * i[j, :].T + ajp * i[j1, :].T + cjm * f[j, :].T + cjp * f[j1, :].T

    return dmt.T


def _absorb_constraints(design_matrix, constraints):
    """Absorb model parameters constraints into the design matrix.

    :param design_matrix: The (2-d array) initial design matrix.
    :param constraints: The 2-d array defining initial model parameters
     (``betas``) constraints (``numpy.dot(constraints, betas) = 0``).
    :return: The new design matrix with absorbed parameters constraints.

    :raise ImportError: if scipy is not found, used for ``scipy.linalg.qr()``
      which is cleaner than numpy's version requiring a call like
      ``qr(..., mode='complete')`` to get a full QR decomposition.
    """
    try:
        from scipy import linalg
    except ImportError:  # pragma: no cover
        raise ImportError("Cubic spline functionality requires scipy.")

    m = constraints.shape[0]
    q, r = linalg.qr(numpy.transpose(constraints))

    return numpy.dot(design_matrix, q[:, m:])


def _get_crs_dmatrix(x, knots, constraints=None, cyclic=False):
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
    dm = _get_free_crs_dmatrix(x, knots, cyclic)
    if constraints is not None:
        dm = _absorb_constraints(dm, constraints)

    return dm


def cyclic_cubic_spline(
        x: Union[pandas.Series, numpy.ndarray],
        df: Optional[int]=None,
        knots: Optional[Iterable[float]] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        constraints=None,
        _state: dict = {},
):
        x_orig = x
        x = numpy.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError(
                "Input to cyclic_cubic_spline must be 1-d or a 2-d column vector."
            )
        dm = _get_crs_dmatrix(
            x, all_knots, constraints, cyclic=True
        )
        if have_pandas:
            if isinstance(x_orig, (pandas.Series, pandas.DataFrame)):
                dm = pandas.DataFrame(dm)
                dm.index = x_orig.index
        return dm



def natural_cubic_spline(n):
    pass

class CubicRegressionSpline(object):
    """Base class for cubic regression spline stateful transforms

    This class contains all the functionality for the following stateful
    transforms:
     - ``cr(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)``
       for natural cubic regression spline
     - ``cc(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)``
       for cyclic cubic regression spline
    """

    common_doc = """
    :arg df: The number of degrees of freedom to use for this spline. The
      return value will have this many columns. You must specify at least one
      of ``df`` and ``knots``.
    :arg knots: The interior knots to use for the spline. If unspecified, then
      equally spaced quantiles of the input data are used. You must specify at
      least one of ``df`` and ``knots``.
    :arg lower_bound: The lower exterior knot location.
    :arg upper_bound: The upper exterior knot location.
    :arg constraints: Either a 2-d array defining general linear constraints
     (that is ``numpy.dot(constraints, betas)`` is zero, where ``betas`` denotes
     the array of *initial* parameters, corresponding to the *initial*
     unconstrained design matrix), or the string
     ``'center'`` indicating that we should apply a centering constraint
     (this constraint will be computed from the input data, remembered and
     re-used for prediction from the fitted model).
     The constraints are absorbed in the resulting design matrix which means
     that the model is actually rewritten in terms of
     *unconstrained* parameters. For more details see :ref:`spline-regression`.

    This is a stateful transforms (for details see
    :ref:`stateful-transforms`). If ``knots``, ``lower_bound``, or
    ``upper_bound`` are not specified, they will be calculated from the data
    and then the chosen values will be remembered and re-used for prediction
    from the fitted model.

    Using this function requires scipy be installed.

    .. versionadded:: 0.3.0
    """

    def __init__(self, name, cyclic):
        self._name = name
        self._cyclic = cyclic
        self._tmp = {}
        self._all_knots = None
        self._constraints = None

    def memorize_chunk(
        self,
        x,
        df=None,
        knots=None,
        lower_bound=None,
        upper_bound=None,
        constraints=None,
    ):
        args = {
            "df": df,
            "knots": knots,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "constraints": constraints,
        }
        self._tmp["args"] = args

        x = numpy.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError(
                "Input to %r must be 1-d, " "or a 2-d column vector." % (self._name,)
            )

        self._tmp.setdefault("xs", []).append(x)

    def memorize_finish(self):
        args = self._tmp["args"]
        xs = self._tmp["xs"]
        # Guards against invalid subsequent memorize_chunk() calls.
        del self._tmp

        x = numpy.concatenate(xs)
        if args["df"] is None and args["knots"] is None:
            raise ValueError("Must specify either 'df' or 'knots'.")

        constraints = args["constraints"]
        n_constraints = 0
        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                # Here we collect only number of constraints,
                # actual centering constraint will be computed after all_knots
                n_constraints = 1
            else:
                constraints = numpy.atleast_2d(constraints)
                if constraints.ndim != 2:
                    raise ValueError("Constraints must be 2-d array or " "1-d vector.")
                n_constraints = constraints.shape[0]

        n_inner_knots = None
        if args["df"] is not None:
            min_df = 1
            if not self._cyclic and n_constraints == 0:
                min_df = 2
            if args["df"] < min_df:
                raise ValueError(
                    "'df'=%r must be greater than or equal to %r."
                    % (args["df"], min_df)
                )
            n_inner_knots = args["df"] - 2 + n_constraints
            if self._cyclic:
                n_inner_knots += 1
        self._all_knots = _get_all_sorted_knots(
            x,
            n_inner_knots=n_inner_knots,
            inner_knots=args["knots"],
            lower_bound=args["lower_bound"],
            upper_bound=args["upper_bound"],
        )
        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                # Now we can compute centering constraints
                constraints = _get_centering_constraint_from_dmatrix(
                    _get_free_crs_dmatrix(x, self._all_knots, cyclic=self._cyclic)
                )

            df_before_constraints = self._all_knots.size
            if self._cyclic:
                df_before_constraints -= 1
            if constraints.shape[1] != df_before_constraints:
                raise ValueError(
                    "Constraints array should have %r columns but"
                    " %r found." % (df_before_constraints, constraints.shape[1])
                )
            self._constraints = constraints

    def transform(
        self,
        x,
        df=None,
        knots=None,
        lower_bound=None,
        upper_bound=None,
        constraints=None,
    ):
        x_orig = x
        x = numpy.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError(
                "Input to %r must be 1-d, " "or a 2-d column vector." % (self._name,)
            )
        dm = _get_crs_dmatrix(
            x, self._all_knots, self._constraints, cyclic=self._cyclic
        )
        if have_pandas:
            if isinstance(x_orig, (pandas.Series, pandas.DataFrame)):
                dm = pandas.DataFrame(dm)
                dm.index = x_orig.index
        return dm

    __getstate__ = no_pickling


class CR(CubicRegressionSpline):
    """cr(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)

    Generates a natural cubic spline basis for ``x``
    (with the option of absorbing centering or more general parameters
    constraints), allowing non-linear fits. The usual usage is something like::

      y ~ 1 + cr(x, df=5, constraints='center')

    to fit ``y`` as a smooth function of ``x``, with 5 degrees of freedom
    given to the smooth, and centering constraint absorbed in
    the resulting design matrix. Note that in this example, due to the centering
    constraint, 6 knots will get computed from the input data ``x``
    to achieve 5 degrees of freedom.


    .. note:: This function reproduce the cubic regression splines 'cr' and 'cs'
      as implemented in the R package 'mgcv' (GAM modelling).

    """

    # Under python -OO, __doc__ will be defined but set to None
    if __doc__:
        __doc__ += CubicRegressionSpline.common_doc

    def __init__(self):
        CubicRegressionSpline.__init__(self, name="cr", cyclic=False)


cr = stateful_transform(CR)


class CC(CubicRegressionSpline):
    """cc(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)

    Generates a cyclic cubic spline basis for ``x``
    (with the option of absorbing centering or more general parameters
    constraints), allowing non-linear fits. The usual usage is something like::

      y ~ 1 + cc(x, df=7, constraints='center')

    to fit ``y`` as a smooth function of ``x``, with 7 degrees of freedom
    given to the smooth, and centering constraint absorbed in
    the resulting design matrix. Note that in this example, due to the centering
    and cyclic constraints, 9 knots will get computed from the input data ``x``
    to achieve 7 degrees of freedom.

    .. note:: This function reproduce the cubic regression splines 'cc'
      as implemented in the R package 'mgcv' (GAM modelling).

    """

    # Under python -OO, __doc__ will be defined but set to None
    if __doc__:
        __doc__ += CubicRegressionSpline.common_doc

    def __init__(self):
        CubicRegressionSpline.__init__(self, name="cc", cyclic=True)


cc = stateful_transform(CC)
