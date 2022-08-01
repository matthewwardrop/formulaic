from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import wrapt

if TYPE_CHECKING:  # pragma: no cover
    from .model_spec import ModelSpec


class ModelMatrix(wrapt.ObjectProxy):
    """
    A wrapper around arbitrary model matrix output representations.

    This wrapper allows for `isinstance(..., ModelMatrix)` checks, and allows
    one to access the `ModelSpec` instance associated with its creation using
    `<model_matrix>.model_spec`. All other instance attributes and methods of
    the wrapped object are directly accessible as if the object were unwrapped.
    """

    def __init__(self, matrix: Any, spec: Optional[ModelSpec] = None):
        wrapt.ObjectProxy.__init__(self, matrix)
        self._self_model_spec = spec

    @property
    def model_spec(self) -> Optional[ModelSpec]:
        """
        The `ModelSpec` instance associated with the creation of this
        `ModelMatrix` instance.

        This `ModelSpec` instance can be used to create other `ModelMatrix`s
        that respect all the choices (including feature selection and encoding)
        that were made in the construction of this `ModelMatrix` instance.

        """
        return self._self_model_spec

    def __repr__(self):
        return self.__wrapped__.__repr__()  # pragma: no cover
