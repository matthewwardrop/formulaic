from .basis_spline import basis_spline
from .center import center
from .identity import identity
from .encode_categorical import encode_categorical


TRANSFORMS = {
    'bs': basis_spline,
    'center': center,
    'C': encode_categorical,
    'I': identity,
}
