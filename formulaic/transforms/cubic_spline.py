from __future__ import annotations
from typing import Optional, Iterable, Union, Literal
import numpy
import pandas
from formulaic.materializers.types import FactorValues
from formulaic.utils.stateful_transforms import stateful_transform
from formulaic.transforms.basis_spline import SplineExtrapolation


def safe_string_eq(obj, value):
    if isinstance(obj, str):
        return obj == value
    else:
        return False

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


def _get_all_sorted_knots(
    x, n_inner_knots=None, inner_knots=None, lower_bound=None, upper_bound=None
):
    """Gets all knots locations with lower and upper exterior knots included.

    If needed, inner knots are computed as equally spaced quantiles of the
    input data falling between given lower and upper bounds.

    :param x: The 1-d array data values.
    :param n_inner_knots: Number of inner knots to compute.
    :param inner_knots: Provided inner knots if any.
    :param lower_bound: The lower exterior knot location. If unspecified, the
     minimum of ``x`` values is used.
    :param upper_bound: The upper exterior knot location. If unspecified, the
     maximum of ``x`` values is used.
    :return: The array of ``n_inner_knots + 2`` distinct knots.

    :raise ValueError: for various invalid parameters sets or if unable to
     compute ``n_inner_knots + 2`` distinct knots.
    """
    if lower_bound is None and x.size == 0:
        raise ValueError(
            "Cannot set lower exterior knot location: empty "
            "input data and lower_bound not specified."
        )
    elif lower_bound is None and x.size != 0:
        lower_bound = numpy.min(x)

    if upper_bound is None and x.size == 0:
        raise ValueError(
            "Cannot set upper exterior knot location: empty "
            "input data and upper_bound not specified."
        )
    elif upper_bound is None and x.size != 0:
        upper_bound = numpy.max(x)

    if upper_bound < lower_bound:
        raise ValueError(
            "lower_bound > upper_bound (%r > %r)" % (lower_bound, upper_bound)
        )

    if inner_knots is None and n_inner_knots is not None:
        if n_inner_knots < 0:
            raise ValueError(
                "Invalid requested number of inner knots: %r" % (n_inner_knots,)
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
                "No data values between lower_bound(=%r) and "
                "upper_bound(=%r): cannot compute requested "
                "%r inner knot(s)." % (lower_bound, upper_bound, n_inner_knots)
            )
    elif inner_knots is not None:
        inner_knots = numpy.unique(inner_knots)
        if n_inner_knots is not None and n_inner_knots != inner_knots.size:
            raise ValueError(
                "Needed number of inner knots=%r does not match "
                "provided number of inner knots=%r." % (n_inner_knots, inner_knots.size)
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

    all_knots = numpy.concatenate(([lower_bound, upper_bound], inner_knots))
    all_knots = numpy.unique(all_knots)
    if all_knots.size != n_inner_knots + 2:
        raise ValueError(
            "Unable to compute n_inner_knots(=%r) + 2 distinct "
            "knots: %r data value(s) found between "
            "lower_bound(=%r) and upper_bound(=%r)."
            % (n_inner_knots, x.size, lower_bound, upper_bound)
        )

    return all_knots

def _get_centering_constraint_from_dmatrix(design_matrix):
    """Computes the centering constraint from the given design matrix.

    We want to ensure that if ``b`` is the array of parameters, our
    model is centered, ie ``numpy.mean(numpy.dot(design_matrix, b))`` is zero.
    We can rewrite this as ``numpy.dot(c, b)`` being zero with ``c`` a 1-row
    constraint matrix containing the mean of each column of ``design_matrix``.

    :param design_matrix: The 2-d array design matrix.
    :return: A 2-d array (1 x ncols(design_matrix)) defining the
     centering constraint.
    """
    return design_matrix.mean(axis=0).reshape((1, design_matrix.shape[1]))

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
    m = constraints.shape[0]
    q, r = numpy.linalg.qr(numpy.transpose(constraints), mode="complete")

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
        if isinstance(x_orig, (pandas.Series, pandas.DataFrame)):
            dm = pandas.DataFrame(dm)
            dm.index = x_orig.index
        return dm


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

@stateful_transform
def cyclic_cubic_spline(  # pylint: disable=dangerous-default-value  # always replaced by stateful-transform
    x: Union[pandas.Series, numpy.ndarray],
    df: Optional[int] = None,
    knots: Optional[Iterable[float]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    constraints: Optional[Union[numpy.ndarray, Literal["center"]]] = None,
    extrapolation: Union[str, SplineExtrapolation] = "raise",
    _state: dict = {},
) -> FactorValues[dict]:
    """
    Evaluates the B-Spline basis vectors for given inputs `x`.

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
        degree: The degree of the B-Spline (the highest degree of terms in the
            resulting polynomial). Must be a non-negative integer.
        include_intercept: Whether to return a complete (full-rank) basis. Note
            that if `ensure_full_rank=True` is passed to the materializer, then
            the intercept will (depending on context) nevertheless be omitted.
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
        lower_bound = _state["lower_bound"]
    else:
        lower_bound = _state["lower_bound"] = (
            numpy.min(x) if lower_bound is None else lower_bound
        )

    if "upper_bound" in _state:
        upper_bound = _state["upper_bound"]
    else:
        upper_bound = _state["upper_bound"] = (
            numpy.max(x) if upper_bound is None else upper_bound
        )

    extrapolation = SplineExtrapolation(extrapolation)

    # Prepare data
    if extrapolation is SplineExtrapolation.RAISE and numpy.any(
        (x < lower_bound) | (x > upper_bound)
    ):
        raise ValueError(
            "Some field values extend beyond upper and/or lower bounds, which can result in ill-conditioned bases. "
            "Pass a value for `extrapolation` to control how extrapolation should be performed."
        )
    if extrapolation is SplineExtrapolation.CLIP:
        x = numpy.clip(x, lower_bound, upper_bound)
    if extrapolation is SplineExtrapolation.NA:
        x = numpy.where((x >= lower_bound) & (x <= upper_bound), x, numpy.nan)

    # Prepare knots
    if "knots" not in _state:
        knots = [] if knots is None else list(knots)
        if df:
            nknots = df - degree - (1 if include_intercept else 0)
            if nknots < 0:
                raise ValueError(
                    f"Invalid value for `df`. `df` must be greater than {degree + (1 if include_intercept else 0)} [`degree` (+ 1 if `include_intercept` is `True`)]."
                )
            knots = list(
                numpy.quantile(x, numpy.linspace(0, 1, nknots + 2))[1:-1].ravel()
            )
        knots.insert(0, cast(float, lower_bound))
        knots.append(cast(float, upper_bound))
        knots = list(numpy.pad(knots, degree, mode="edge"))
        _state["knots"] = knots
    knots = _state["knots"]

    # Compute basis splines
    # The following code is equivalent to [B(i, j=degree) for in range(len(knots)-d-1)], with B(i, j) as defined below.
    # B = lambda i, j: ((x >= knots[i]) & (x < knots[i+1])).astype(float) if j == 0 else alpha(i, j, x) * B(i, j-1, x) + (1 - alpha(i+1, j, x)) * B(i+1, j-1, x)
    # We don't directly use this recurrence relation so that we can memoise the B(i, j).
    cache: Dict[int, Dict[int, float]] = defaultdict(dict)
    alpha = (
        lambda i, j: (x - knots[i]) / (knots[i + j] - knots[i])
        if knots[i + j] != knots[i]
        else 0
    )
    for i in range(len(knots) - 1):
        if extrapolation is SplineExtrapolation.EXTEND:
            cache[0][i] = (
                (x >= (knots[i] if i != degree else -numpy.inf))
                & (
                    x
                    < (knots[i + 1] if i + 1 != len(knots) - degree - 1 else numpy.inf)
                )
            ).astype(float)
        else:
            cache[0][i] = (
                (x >= knots[i])
                & (
                    (x < knots[i + 1])
                    if i + 1 != len(knots) - degree - 1
                    else (x <= knots[i + 1])  # Properly handle boundary
                )
            ).astype(float)
    for d in range(1, degree + 1):
        cache[d % 2].clear()
        for i in range(len(knots) - d - 1):
            cache[d % 2][i] = (
                alpha(i, d) * cache[(d - 1) % 2][i]
                + (1 - alpha(i + 1, d)) * cache[(d - 1) % 2][i + 1]
            )

    return FactorValues(
        {
            i: cache[degree % 2][i]
            for i in sorted(cache[degree % 2])
            if i > 0 or include_intercept
        },
        kind="numerical",
        spans_intercept=include_intercept,
        drop_field=0,
        format="{name}[{field}]",
        encoded=False,
    )
