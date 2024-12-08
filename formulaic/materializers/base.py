# pragma: no cover
import warnings

from .types import FormulaMaterializer

warnings.warn(
    "`FormulaMaterializer` has been moved from `formulaic.materializers.base` to `formulaic.materializers.types.formula_materializer`. This shim will be removed in version 2.0.",
    DeprecationWarning,
)

__all__ = ["FormulaMaterializer"]
