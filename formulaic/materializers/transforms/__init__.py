from .center import center
from .encode_categorical import encode_categorical


TRANSFORMS = {
    'center': center,
    'C': encode_categorical,
}
