from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, MutableSet, Sequence
from itertools import islice
from typing import Any, Generic, TypeVar, Union, overload

_ItemType = TypeVar("_ItemType")
_SelfType = TypeVar("_SelfType", bound="OrderedSet")


class OrderedSet(MutableSet, Sequence, Generic[_ItemType]):
    """
    A mutable set-like sequenced container that retains the order in which item
    were added to the set, keeps track of multiplicities (how many times an item
    was added), and provides both set and list-like indexing and mutations. This
    container keeps track of how many times an item was added to the set, which
    can be checked using the `.get_multiplicity()` method.
    """

    def __init__(
        self,
        values: Union[
            Iterable[_ItemType], Mapping[_ItemType, int], OrderedSet[_ItemType]
        ] = (),
    ) -> None:
        self._values: Counter = Counter(
            values._values if isinstance(values, OrderedSet) else values
        )

    def get_multiplicity(self, item: _ItemType) -> int:
        """
        Identify how many times this item was added to the set. If the item was
        never added, return 0. This is mainly useful if you later need to expand
        an item into multiple items and need to keep track of the original
        interaction order.
        """
        return self._values[item]

    def __repr__(self) -> str:
        return f"{{{', '.join(repr(v) for v in self._values)}}}"

    # MutableSet interface

    def __contains__(self, item: Any) -> bool:
        return item in self._values

    def __iter__(self) -> Iterator[_ItemType]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, item: _ItemType) -> None:
        self._values.update((item,))

    def discard(self, item: _ItemType) -> None:
        if item in self._values:
            del self._values[item]

    # Additional methods for Sequence interface (O(n) lookups by index)

    @overload
    def __getitem__(self, index: int) -> _ItemType: ...

    @overload
    def __getitem__(self: _SelfType, index: slice) -> _SelfType: ...

    def __getitem__(
        self: _SelfType, index: Union[int, slice]
    ) -> Union[_ItemType, _SelfType]:
        if isinstance(index, slice):
            return self.__class__(
                {
                    item: self._values[item]
                    for item in islice(
                        self._values, index.start, index.stop, index.step
                    )
                }
            )
        else:
            return next(islice(self._values, index, None))

    # Convenience methods

    def update(
        self,
        items: Union[
            Iterable[_ItemType], Mapping[_ItemType, int], OrderedSet[_ItemType]
        ],
    ) -> None:
        """
        Update this ordered set with the items from another iterable or mapping
        from items to observed counts.

        Args:
            items: The items to add to this ordered set. If an iterable is
                is provided, the items will be added with a count of 1.
                Otherwise the counts will be aggregated from the mapping and/or
                ordered set instances.
        """
        self._values.update(items._values if isinstance(items, OrderedSet) else items)
