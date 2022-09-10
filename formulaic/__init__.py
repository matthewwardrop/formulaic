from .formula import Formula, FormulaSpec
from .materializers import FactorValues
from .model_matrix import ModelMatrix, ModelMatrices
from .model_spec import ModelSpec, ModelSpecs
from .sugar import model_matrix

try:
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    __version__ = version = "unknown"
    __version_tuple__ = version_tuple = ("unknown",)

__author__ = "Matthew Wardrop"
__author_email__ = "mpwardrop@gmail.com"

__all__ = [
    "__author__",
    "__author_email__",
    "__version__",
    "__version_tuple__",
    "Formula",
    "FormulaSpec",
    "ModelMatrix",
    "ModelMatrices",
    "ModelSpec",
    "ModelSpecs",
    "model_matrix",
    "FactorValues",
]
