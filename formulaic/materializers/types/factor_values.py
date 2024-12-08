from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

try:
    from typing import SupportsIndex
except ImportError:  # pragma: no cover
    from typing_extensions import SupportsIndex

import wrapt

from formulaic.parser.types import Factor
from formulaic.utils.sentinels import MISSING, MissingType

if TYPE_CHECKING:  # pragma: no cover
    from formulaic.model_spec import ModelSpec


T = TypeVar("T")


@dataclass
class FactorValuesMetadata:
    """
    Metadata about evaluated factor values.

    This metadata is used to inform materializers about how to treat these
    values.

    Attributes:
        kind: The kind of the evaluated values.
        format: The format to use when exploding factors into multiple columns
            (e.g. when encoding categories via dummy-encoding).
        encoded: Whether the values should be treated as pre-encoded.
        encoder: An optional callable with signature
            `(values: Any, reduced_rank: bool, drop_rows: List[int], encoder_state: Dict[str, Any], spec: ModelSpec)`
            that outputs properly encoded values suitable for the current
            materializer. Note that this should only be used in cases where
            direct evaluation would yield different results in reduced vs.
            non-reduced rank scenarios.

        Rank-Reduction Attributes:
            spans_intercept: Whether the values span the intercept or not.
            drop_field: If the values do span the intercept, and we want to reduce
                the rank, which field should be dropped.
            reduced: Whether the rank has already been reduced by dropping the
                `drop_field` above.
            format_reduced: The format to use when exploding factors (as above), but
                in the case where the rank has been reduced by dropping a field.
                (This defaults to `format`.)
    """

    kind: Factor.Kind = Factor.Kind.UNKNOWN
    column_names: Optional[Tuple[str]] = None
    format: str = "{name}[{field}]"
    encoded: bool = False
    encoder: Optional[
        Callable[[Any, bool, List[int], Dict[str, Any], ModelSpec], Any]
    ] = None

    # Rank-Reduction Attributes
    spans_intercept: bool = False
    drop_field: Optional[str] = None
    reduced: bool = False
    format_reduced: Optional[str] = None

    def get_format(self) -> str:
        return (
            self.format_reduced if self.reduced and self.format_reduced else self.format
        )

    def replace(self, **kwargs: Any) -> FactorValuesMetadata:
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
        metadata: Union[FactorValuesMetadata, MissingType] = MISSING,
        *,
        kind: Union[str, Factor.Kind, MissingType] = MISSING,
        column_names: Union[Tuple[Hashable, ...], MissingType] = MISSING,
        format: Union[str, MissingType] = MISSING,  # pylint: disable=redefined-builtin
        encoded: Union[bool, MissingType] = MISSING,
        encoder: Union[
            None,
            Callable[[Any, bool, List[int], Dict[str, Any], ModelSpec], Any],
            MissingType,
        ] = MISSING,
        spans_intercept: Union[bool, MissingType] = MISSING,
        drop_field: Union[None, Hashable, MissingType] = MISSING,
        reduced: Union[bool, MissingType] = MISSING,
        format_reduced: Union[str, MissingType] = MISSING,
    ):
        metadata_constructor: Callable = FactorValuesMetadata
        metadata_kwargs = dict(
            kind=Factor.Kind(kind) if kind is not MISSING else kind,
            column_names=column_names,
            format=format,
            encoded=encoded,
            encoder=encoder,
            spans_intercept=spans_intercept,
            drop_field=drop_field,
            reduced=reduced,
            format_reduced=format_reduced,
        )
        for key in set(metadata_kwargs):
            if metadata_kwargs[key] is MISSING:
                metadata_kwargs.pop(key)

        if hasattr(values, "__formulaic_metadata__"):
            metadata_constructor = values.__formulaic_metadata__.replace
            if isinstance(values, FactorValues):
                values = values.__wrapped__

        if metadata and metadata is not MISSING:
            metadata_constructor = metadata.replace

        wrapt.ObjectProxy.__init__(self, values)
        self._self_metadata = metadata_constructor(**metadata_kwargs)

    @property
    def __formulaic_metadata__(self) -> FactorValuesMetadata:
        return self._self_metadata

    def __repr__(self) -> str:
        return self.__wrapped__.__repr__()  # pragma: no cover

    # Handle copying behaviour

    def __copy__(self) -> FactorValues[T]:
        return type(self)(copy.copy(self.__wrapped__), metadata=self._self_metadata)

    def __deepcopy__(self, memo: Any = None) -> FactorValues[T]:
        return type(self)(
            copy.deepcopy(self.__wrapped__, memo),
            metadata=copy.deepcopy(self._self_metadata),
        )

    # Handle pickling behaviour

    def __reduce_ex__(
        self, protocol: SupportsIndex
    ) -> Tuple[
        Callable[[Any, Union[FactorValuesMetadata, MissingType]], FactorValues],
        Tuple[Any, Union[FactorValuesMetadata, MissingType]],
    ]:
        return FactorValues, (self.__wrapped__, self._self_metadata)
