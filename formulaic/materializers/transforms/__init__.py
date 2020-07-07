from .center import center
from .identity import identity
from .encode_categorical import encode_categorical


TRANSFORMS = {
    'center': center,
    'C': encode_categorical,
    'I': identity,
}
