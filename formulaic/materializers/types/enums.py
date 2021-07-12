from enum import Enum


class NAAction(Enum):
    DROP = "drop"
    RAISE = "raise"
    IGNORE = "ignore"
