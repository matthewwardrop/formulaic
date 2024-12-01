from .formula import Formula, FormulaSpec, SimpleFormula, StructuredFormula
from .materializers import FactorValues
from .model_matrix import ModelMatrices, ModelMatrix
from .model_spec import ModelSpec, ModelSpecs
from .sugar import model_matrix

try:
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    __version__ = version = "unknown"
    __version_tuple__ = version_tuple = ("unknown",)  # type: ignore

__author__ = "Matthew Wardrop"
__author_email__ = "mpwardrop@gmail.com"

__all__ = [
    "__author__",
    "__author_email__",
    "__version__",
    "__version_tuple__",
    "Formula",
    "SimpleFormula",
    "StructuredFormula",
    "FormulaSpec",
    "ModelMatrix",
    "ModelMatrices",
    "ModelSpec",
    "ModelSpecs",
    "model_matrix",
    "FactorValues",
]
