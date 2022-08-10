from __future__ import annotations
import copy

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import wrapt

from formulaic.parser.types import Factor
from formulaic.utils.sentinels import MISSING


T = TypeVar("T")


@dataclass
class FactorValuesMetadata:
    """
    Metadata about evaluated factor values.

    This metadata is used to inform materializers about how to treat these
    values.

    Attributes:
        kind: The kind of the evaluated values.
        spans_intercept: Whether the values span the intercept or not.
        drop_field: If the values do span the intercept, and we want to reduce
            the rank, which field should be dropped.
        format: The format to use when exploding factors into multiple columns
            (e.g. when encoding categories via dummy-encoding).
        encoded: Whether the values should be treated as pre-encoded.
        encoder: An optional callable with signature
            `(values: Any, reduced_rank: bool, drop_rows: List[int], encoder_state: Dict[str, Any], spec: ModelSpec)`
            that outputs properly encoded values suitable for the current
            materializer. Note that this should only be used in cases where
            direct evaluation would yield different results in reduced vs.
            non-reduced rank scenarios.
    """

    kind: Factor.Kind = Factor.Kind.UNKNOWN
    column_names: Optional[Tuple[str]] = None
    spans_intercept: bool = False
    drop_field: Optional[str] = None
    format: str = "{name}[{field}]"
    encoded: bool = False
    encoder: Optional[Callable[[Any, bool, List[int], Dict[str, Any]], Any]] = None

    def replace(self, **kwargs) -> FactorValuesMetadata:
        """
        Return a copy of this `FactorValuesMetadata` instance with the nominated
        attributes replaced.
        """
        if not kwargs:
            return self
        return replace(self, **kwargs)


class FactorValues(Generic[T], wrapt.ObjectProxy):
    """
    A convenience wrapper that surfaces a `FactorValuesMetadata` instance at
    `<object>.__formulaic_metadata__`. This wrapper can otherwise wrap any
    object and behaves just like that object.
    """

    def __init__(
        self,
        values: Any,
        *,
        metadata: FactorValuesMetadata = MISSING,
        kind: Union[str, Factor.Kind] = MISSING,
        column_names: Tuple[str] = MISSING,
        spans_intercept: bool = MISSING,
        drop_field: Optional[str] = MISSING,
        format: str = MISSING,  # pylint: disable=redefined-builtin
        encoded: bool = MISSING,
        encoder: Optional[
            Callable[[Any, bool, List[int], Dict[str, Any]], Any]
        ] = MISSING,
    ):
        metadata_constructor = FactorValuesMetadata
        metadata_kwargs = dict(
            kind=Factor.Kind(kind) if kind is not MISSING else kind,
            column_names=column_names,
            spans_intercept=spans_intercept,
            drop_field=drop_field,
            format=format,
            encoded=encoded,
            encoder=encoder,
        )
        for key in set(metadata_kwargs):
            if metadata_kwargs[key] is MISSING:
                metadata_kwargs.pop(key)

        if hasattr(values, "__formulaic_metadata__"):
            metadata_constructor = values.__formulaic_metadata__.replace
            if isinstance(values, FactorValues):
                values = values.__wrapped__

        if metadata:
            metadata_constructor = metadata.replace

        wrapt.ObjectProxy.__init__(self, values)
        self._self_metadata = metadata_constructor(**metadata_kwargs)

    @property
    def __formulaic_metadata__(self) -> FactorValuesMetadata:
        return self._self_metadata

    def __repr__(self) -> str:
        return self.__wrapped__.__repr__()  # pragma: no cover

    # Handle copying behaviour

    def __copy__(self):
        return type(self)(copy.copy(self.__wrapped__), metadata=self._self_metadata)

    def __deepcopy__(self, memo=None):
        return type(self)(
            copy.deepcopy(self.__wrapped__, memo),
            metadata=copy.deepcopy(self._self_metadata),
        )
