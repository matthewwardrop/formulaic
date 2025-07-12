from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, MutableSequence, MutableSet
from itertools import chain, islice
from typing import Any, Generic, Optional, TypeVar, Union, overload

_ItemType = TypeVar("_ItemType")
_SelfType = TypeVar("_SelfType", bound="OrderedSet")


class OrderedSet(MutableSet, MutableSequence, Generic[_ItemType]):
    """
    A mutable set-like sequence container that retains the order in which item
    were added to the set, keeps track of multiplicities (how many times an item
    was added), and provides both set and list-like indexing and mutations. This
    container keeps track of how many times an item was added to the set, which
    can be checked using the `.get_multiplicity()` method.

    This class is optimised for set-like operations, but also provides O(n)
    lookups by index, insertions, deletions, and updates. We may optimise this
    in the future based on need by maintaining index tables.

    Note: Indexed mutations like `collection[<index>] = <value>` do not maintain
    order, and are equivalent to `collection.remove(collection[<index>])`
    followed by `collection.add(<value>)`.
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

    def _prepare_item(self, item: Any) -> _ItemType:
        """
        Prepare an item for insertion into this ordered set. This method is
        called whenever an item is added to the set. It is *not* called for
        discard operations.
        """
        return item

    def _post_update(self) -> None:
        """
        Perform any post-update operations. This is called after every mutation
        to the ordered set.
        """
        pass

    def __repr__(self) -> str:
        return f"{{{', '.join(repr(v) for v in self._values)}}}"

    # MutableSet interface

    def __contains__(self, item: Any) -> bool:
        return item in self._values

    def __iter__(self) -> Iterator[_ItemType]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, item: _ItemType, count: int = 1) -> None:
        item = self._prepare_item(item)
        self._values[item] = self._values.get(item, 0) + count
        self._post_update()

    def discard(self, item: _ItemType, count: Optional[int] = None) -> None:
        if item in self._values:
            final_count = 0 if count is None else self._values.get(item) - count
            if final_count <= 0:
                del self._values[item]
            else:
                self._values[item] = final_count
        self._post_update()

    # MutableSet order preservation

    def __and__(self, other: Any) -> OrderedSet[_ItemType]:
        out = OrderedSet[_ItemType]()
        other = OrderedSet(other)

        for value, count in self._values.items():
            if value in other:
                out.add(value, count)
        for value, count in other._values.items():
            if value in self:
                out.add(value, count)
        return out

    def __ror__(self, other: Any) -> OrderedSet[_ItemType]:
        return OrderedSet(other) | self

    def __rxor__(self, other: Any) -> OrderedSet[_ItemType]:
        return OrderedSet(other) ^ self

    def __rand__(self, other: Any) -> OrderedSet[_ItemType]:
        return OrderedSet(other) & self

    def __rsub__(self, other: Any) -> OrderedSet[_ItemType]:
        return OrderedSet(other) - self

    # Additional methods for MutableSequence interface (O(n) lookups by index)

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
                        self._values,
                        index.start % len(self) if index.start is not None else None,
                        index.stop % len(self) if index.stop is not None else None,
                        index.step,
                    )
                }
            )
        else:
            return next(islice(self._values, index % len(self), None))

    @overload
    def __setitem__(self, key: int, value: _ItemType) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[_ItemType]) -> None: ...

    def __setitem__(self, key, value):  # type: ignore
        self.__insert_or_replace(
            key, value, replace=True
        )

    @overload
    def __delitem__(self, key: int) -> None: ...

    @overload
    def __delitem__(self, key: slice) -> None: ...

    def __delitem__(self, key):  # type: ignore
        if isinstance(key, slice):
            for item in self[key]:
                del self._values[item]
        else:
            del self._values[self[key]]
        self._post_update()

    def insert(self, index: int, value: _ItemType, count: int = 1) -> None:
        self.__insert_or_replace(index, value, count=count, replace=False)

    def __insert_or_replace(self, indices: Union[int, slice], values: Union[_ItemType, Iterable[_ItemType]], count: int = 1, replace=False) -> None:
        if isinstance(indices, int):
            indices = range(indices, indices+1)
            values = [values]
        else:
            indices = range(len(self._values))[indices]
        if len(indices) == 0:
            indices = range(len(self._values), len(self._values) + 1)

        values_to_insert = {
            v: self._values.get(v, 0) + count
            for value in values
            if (v := self._prepare_item(value)) or True
        }
        _values_new = Counter()
        for i, (item, count) in enumerate(self._values.items()):
            if i in indices:
                if i > min(indices):
                    continue
                _values_new.update(values_to_insert)
            if item not in values_to_insert and (not replace or i not in indices):
                _values_new.update({item: count})
        if min(indices) >= len(self._values):
            _values_new.update(values_to_insert)
        self._values = _values_new
        self._post_update()

    # Other data model methods

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (OrderedSet, list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

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
        self._values.update(items.values if isinstance(items, OrderedSet) else items)
        self._post_update()
