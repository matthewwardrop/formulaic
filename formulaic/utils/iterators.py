from typing import Any, Iterable, Iterator, List

from .sentinels import MISSING


class peekable_iter:
    """
    An iterator that allows you to peek at the next element during iteration.
    """

    def __init__(self, it: Iterable):
        self._it = iter(it)
        self._next: List[Any] = []

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Any:
        if self._next:
            return self._next.pop(0)
        return next(self._it)

    def peek(self, default: Any = MISSING) -> Any:
        """
        Retrieve the object that will be next returned by the iterator.

        Args:
            default: The value to return if there are no more elements in the
                iterator (otherwise the `StopIteration` exception will be
                forwarded).
        """
        try:
            if not self._next:
                self._next.append(next(self._it))
            return self._next[0]
        except StopIteration:
            if default is MISSING:
                raise
            return default
