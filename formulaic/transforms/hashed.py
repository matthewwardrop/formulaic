from __future__ import annotations

import sys
from hashlib import md5
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from formulaic.materializers.types import FactorValues

from .contrasts import Contrasts, encode_contrasts

if TYPE_CHECKING:  # pragma: no cover
    from formulaic.model_spec import ModelSpec


def md5_to_int(s: str) -> int:  # pragma: no cover; branched code
    if sys.version_info >= (3, 9):
        hashed = md5(s.encode(), usedforsecurity=False)
    else:
        hashed = md5(s.encode())  # noqa: S324 ; use of insecure hash function
    return int(hashed.hexdigest(), 16)


def hashed(
    data: Any,
    levels: int,
    contrasts: Optional[
        Union[Contrasts, Dict[str, Iterable[Number]], np.ndarray]
    ] = None,
    *,
    hash_func: Callable[[str], int] = md5_to_int,
    spans_intercept: bool = False,
) -> FactorValues:
    """
    Deterministically hashes the values of a factor into a fixed number of levels.
    If `levels` is large, you will likely want to use this transform in conjunction
    with `output='sparse'` to avoid memory issues.

    Args:
        data: The data to feature hash.
        levels: The number of levels to hash into. This should be set to a
            large number to avoid collisions. Per the Birthday Paradox, the
            number of unique feature values that can be hashed with no more
            than a 50% probability of collision is approximately sqrt(2 * levels).
        contrasts: The contrasts to use for this factor. If not provided, the
            `Treatment` contrasts will be used.
        hash_func: The function to use to hash the values. This should return
            an integer. The default is to use the MD5 hash function.
        spans_intercept: Whether the values span the intercept or not, default
            False.

    """

    def encoder(
        values: Any,
        reduced_rank: bool,
        drop_rows: List[int],
        encoder_state: Dict[str, Any],
        model_spec: ModelSpec,
    ) -> FactorValues:
        values = np.array(values)
        return encode_contrasts(
            values,
            contrasts=contrasts,
            levels=np.arange(levels),
            reduced_rank=reduced_rank,
            _spec=model_spec,
        )

    stringified_data = np.array(data).astype(np.str_)

    return FactorValues(
        (np.vectorize(md5_to_int)(stringified_data) % levels).astype(np.int_),
        kind="categorical",
        spans_intercept=spans_intercept,
        encoder=encoder,
    )
