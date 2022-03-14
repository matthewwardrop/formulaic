import numpy

from .basis_spline import basis_spline
from .identity import identity
from .encode_categorical import encode_categorical
from .poly import poly
from .scale import center, scale


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
    "center": center,
    "poly": poly,
    "scale": scale,
    "C": encode_categorical,
    "I": identity,
}
