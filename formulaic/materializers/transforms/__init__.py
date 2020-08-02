from .basis_spline import basis_spline
from .identity import identity
from .encode_categorical import encode_categorical
from .scale import center, scale


TRANSFORMS = {
    'bs': basis_spline,
    'center': center,
    'scale': scale,
    'C': encode_categorical,
    'I': identity,
}
