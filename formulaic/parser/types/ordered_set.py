from collections.abc import Set

from typing import Generic, Iterable, TypeVar

ItemType = TypeVar("ItemType")


class OrderedSet(Set, Generic[ItemType]):
    def __init__(self, values: Iterable[ItemType] = ()):
        self.values = dict.fromkeys(values)

    def __contains__(self, item):
        return item in self.values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"{{{', '.join(repr(v) for v in self.values)}}}"
