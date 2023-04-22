from enum import Enum


class NAAction(Enum):
    DROP = "drop"
    RAISE = "raise"
    IGNORE = "ignore"


class ClusterBy(Enum):
    NONE = "none"
    NUMERICAL_FACTORS = "numerical_factors"
