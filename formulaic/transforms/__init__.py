import numpy

from formulaic.utils.stateful_transforms import stateful_transform

from .basis_spline import basis_spline
from .contrasts import C, ContrastsRegistry, encode_contrasts
from .cubic_spline import cyclic_cubic_spline, natural_cubic_spline
from .hashed import hashed
from .identity import identity
from .lag import lag
from .patsy_compat import PATSY_COMPAT_TRANSFORMS
from .poly import poly
from .scale import center, scale

__all__ = [
    "basis_spline",
    "cubic_spline",
    "cyclic_cubic_spline",
    "natural_cubic_spline",
    "identity",
    "C",
    "encode_contrasts",
    "ContrastsRegistry",
    "lag",
    "poly",
    "center",
    "scale",
    "stateful_transform",
    "TRANSFORMS",
]

TRANSFORMS = {
    # Common transforms
    "np": numpy,
    "log": numpy.log,
    "log10": numpy.log10,
    "log2": numpy.log2,
    "exp": numpy.exp,
    "exp10": lambda x: numpy.power(x, 10),
    "exp2": numpy.exp2,
    # Bespoke transforms
    "bs": basis_spline,
    "cc": cyclic_cubic_spline,
    "cr": natural_cubic_spline,
    "cs": natural_cubic_spline,
    "center": center,
    "lag": lag,
    "poly": poly,
    "scale": scale,
    "C": C,
    "contr": ContrastsRegistry,
    "I": identity,
    "hashed": hashed,
    # Patsy compatibility shims
    **PATSY_COMPAT_TRANSFORMS,
}
