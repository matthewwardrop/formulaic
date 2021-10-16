from .basis_spline import basis_spline
from .identity import identity
from .encode_categorical import encode_categorical
from .poly import poly
from .scale import center, scale


TRANSFORMS = {
    "bs": basis_spline,
    "center": center,
    "poly": poly,
    "scale": scale,
    "C": encode_categorical,
    "I": identity,
}
