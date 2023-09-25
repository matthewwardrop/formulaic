from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from formulaic.materializers.types import FactorValues
from formulaic.utils.stateful_transforms import stateful_transform

try:
    import numpy.typing
except ImportError as e:  # pragma: no cover
    if TYPE_CHECKING:
        raise RuntimeError("Numpy >=1.20 is required for type-checking.") from e


@stateful_transform
def poly(  # pylint: disable=dangerous-default-value  # always replaced by stateful-transform
    x: numpy.typing.ArrayLike, degree: int = 1, raw: bool = False, _state: dict = {}
) -> numpy.ndarray:
    """
    Generate a basis for a polynomial vector-space representation of `x`.

    The basis vectors returned by this transform can be used, for example, to
    capture non-linear dependence on `x` in a linear regression.

    Args:
        x: The vector for which a polynomial vector space should be generated.
        degree: The degree of the polynomial vector space.
        raw: Whether to return "raw" basis vectors (e.g. `[x, x**2, x**3]`). If
            `False`, an orthonormal set of basis vectors is returned instead
            (see notes below for more information).

    Returns:
        A two-dimensional numpy array with `len(x)` rows, and `degree` columns.
        The columns represent the basis vectors of the polynomial vector-space.

    Notes:
        This transform is an implementation of the "three-term recurrence
        relation" for monic orthogonal polynomials. There are many good
        introductions to these recurrence relations, including:
            https://dec41.user.srcf.net/h/IB_L/numerical_analysis/2_3
        Another common approach is QR factorisation, where the columns of Q are
        the orthogonal basis vectors. However, our implementation outperforms
        numpy's QR decomposition, and does not require needless computation of
        the R matrix. It should also be noted that orthogonal polynomial bases
        are unique up to the choice of inner-product and scaling, and so all
        methods will result in the same set of polynomials.

        When used as a stateful transform, we retain the coefficients that
        uniquely define the polynomials; and so new data will be evaluated
        against the same polynomial bases as the original dataset. However,
        the polynomial basis will almost certainly *not* be orthogonal for the
        new data. This is because changing the incoming dataset is equivalent to
        changing your choice of inner product.

        Using orthogonal basis vectors (as compared to the "raw" vectors) allows
        you to increase the degree of the polynomial vector space without
        affecting the coefficients of lower-order components in a linear
        regression. This stability is often attractive during exploratory data
        analysis, but does not otherwise change the results of a linear
        regression.

        `nan` values in `x` will be ignored and progagated through to generated
        polynomials in the corresponding row indices.

        The signature of this transform is intentionally chosen to be compatible
        with R.
    """

    if raw:
        return numpy.stack([numpy.power(x, k) for k in range(1, degree + 1)], axis=1)

    x = numpy.array(x, dtype=numpy.float64)

    # Prepare output matrix (needed because we resize x below if there are nulls)
    out = numpy.empty((x.shape[0], degree))
    out.fill(numpy.nan)

    # Filter to non-null indices (we will invert this later)
    nonnull_indices = numpy.flatnonzero(~numpy.isnan(x))
    x = x[nonnull_indices]

    # Check if we already have generated the alpha and beta coefficients.
    # If not, we enter "training" mode.
    training = False
    alpha = _state.get("alpha")
    norms2 = _state.get("norms2")

    if alpha is None:
        training = True
        alpha = {}
        norms2 = {}

    # Build polynomials iteratively using the monic three-term recurrence relation
    # Note that alpha and beta are fixed if not in "training" mode.
    P = numpy.empty((x.shape[0], degree + 1))
    P[:, 0] = 1

    def get_alpha(k: int) -> float:
        if training and k not in alpha:
            alpha[k] = numpy.sum(x * P[:, k] ** 2) / numpy.sum(P[:, k] ** 2)  # type: ignore[operator]
        return alpha[k]

    def get_norm(k: int) -> float:
        if training and k not in norms2:  # type: ignore[operator]
            norms2[k] = numpy.sum(P[:, k] ** 2)  # type: ignore[index]
        return norms2[k]  # type: ignore[index]

    def get_beta(k: int) -> float:
        return get_norm(k) / get_norm(k - 1)

    for i in range(1, degree + 1):
        P[:, i] = (x - get_alpha(i - 1)) * P[:, i - 1]
        if i >= 2:
            P[:, i] -= get_beta(i - 1) * P[:, i - 2]

    # Renormalize so we provide an orthonormal basis.
    P /= numpy.array([numpy.sqrt(get_norm(k)) for k in range(0, degree + 1)])

    if training:
        _state["alpha"] = alpha
        _state["norms2"] = norms2

    # Populate output matrix with evaluated polynomials
    out[nonnull_indices, :] = P[:, 1:]

    # Return basis dropping the first (constant) column
    return FactorValues(out, column_names=tuple(str(i) for i in range(1, degree + 1)))
