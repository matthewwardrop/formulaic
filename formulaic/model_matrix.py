from __future__ import annotations

import copy
from typing import Any, Generic, Optional, TypeVar, TYPE_CHECKING

import wrapt

from formulaic.parser.types.structured import Structured

if TYPE_CHECKING:  # pragma: no cover
    from .model_spec import ModelSpec, ModelSpecs


MatrixType = TypeVar("MatrixType")


class ModelMatrix(Generic[MatrixType], wrapt.ObjectProxy):
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

    # Handle copying behaviour

    def __copy__(self):
        return type(self)(copy.copy(self.__wrapped__), spec=self._self_model_spec)

    def __deepcopy__(self, memo=None):
        return type(self)(
            copy.deepcopy(self.__wrapped__, memo),
            spec=copy.deepcopy(self._self_model_spec),
        )


class ModelMatrices(Structured[ModelMatrix]):
    """
    A `Structured[ModelMatrix]` subclass that adds a `.model_spec` attribute
    (mirrorin `ModelMatrix.model_spec`) that returns a structured container for
    all the `ModelSpec` instances associated with the `ModelSpec` objects
    referenced by this container.
    """

    def _prepare_item(
        self, key: str, item: Any
    ) -> Any:  # Verify that all included items are `ModelSpec` instances.
        # Verify that all included items are `ModelMatrix` instances.
        if not isinstance(item, ModelMatrix):
            raise TypeError(
                "`ModelMatrices` instances expect all items to be instances "
                f"of `ModelMatrix`. [Got: {repr(item)} of type "
                f"{repr(type(item))} for key {repr(key)}."
            )
        return item

    @property
    def model_spec(self) -> ModelSpecs:
        """
        The `ModelSpecs` instance representing the structured set of `ModelSpec`
        instances associated with the `ModelMatrix` instances stored in this
        `Structured` instance.
        """
        from .model_spec import ModelSpecs

        return self._map(
            lambda model_matrix: model_matrix.model_spec, as_type=ModelSpecs
        )
