from __future__ import annotations

from typing import Dict
from typing_extensions import Self


class _MissingType:
    __instance__ = None

    def __new__(cls) -> _MissingType:
        if cls.__instance__ is None:
            cls.__instance__ = super(_MissingType, cls).__new__(cls)
        return cls.__instance__

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: Dict) -> Self:
        return self


MISSING = _MissingType()
