from __future__ import annotations

from collections.abc import Set
from typing import Any, Generic, Iterable, Iterator, TypeVar

ItemType = TypeVar("ItemType")


class OrderedSet(Set, Generic[ItemType]):
    """
    A set-like container that retains the order in which item were added to the
    set.
    """

    def __init__(self, values: Iterable[ItemType] = ()) -> None:
        self.values = dict.fromkeys(values)

    def __contains__(self, item: Any) -> bool:
        return item in self.values

    def __iter__(self) -> Iterator[ItemType]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"{{{', '.join(repr(v) for v in self.values)}}}"
