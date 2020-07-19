from .center import center
from .identity import identity
from .encode_categorical import encode_categorical
from .basis_splines import basis_splines

TRANSFORMS = {
    'center': center,
    'C': encode_categorical,
    'I': identity,
    'bs': basis_splines,
}
