from __future__ import annotations

from enum import Enum

from typing_extensions import Literal, TypeAlias


class Sentinel(Enum):
    MISSING = "MISSING"
    UNSET = "UNSET"

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return self.name


UnsetType: TypeAlias = Literal[Sentinel.UNSET]
UNSET: UnsetType = Sentinel.UNSET

MissingType: TypeAlias = Literal[Sentinel.MISSING]
MISSING: MissingType = Sentinel.MISSING
