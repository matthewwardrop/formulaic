from collections import defaultdict
from enum import Enum
from typing import Iterable, Optional, Union

import numpy
import pandas

from formulaic.materializers.types import FactorValues
from formulaic.utils.stateful_transforms import stateful_transform


class SplineExtrapolation(Enum):
    """
    Specification for how extrapolation should be performed during spline
    computations.
    """

    RAISE = "raise"
    CLIP = "clip"
    NA = "na"
    ZERO = "zero"
    EXTEND = "extend"


@stateful_transform
def basis_spline(
    x: Union[pandas.Series, numpy.ndarray],
    df: Optional[int] = None,
    knots: Optional[Iterable[float]] = None,
    degree: int = 3,
    include_intercept: bool = False,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    extrapolation: Union[str, SplineExtrapolation] = "raise",
    _state: dict = None,
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
        knots.insert(0, lower_bound)
        knots.append(upper_bound)
        knots = list(numpy.pad(knots, degree, mode="edge"))
        _state["knots"] = knots
    knots = _state["knots"]

    # Compute basis splines
    # The following code is equivalent to [B(i, j=degree) for in range(len(knots)-d-1)], with B(i, j) as defined below.
    # B = lambda i, j: ((x >= knots[i]) & (x < knots[i+1])).astype(float) if j == 0 else alpha(i, j, x) * B(i, j-1, x) + (1 - alpha(i+1, j, x)) * B(i+1, j-1, x)
    # We don't directly use this recurrence relation so that we can memoise the B(i, j).
    cache = defaultdict(dict)
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
