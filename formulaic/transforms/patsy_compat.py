from typing import Any, Dict, Mapping, Optional

from formulaic.utils.stateful_transforms import stateful_transform

from .contrasts import (
    UNSET,
    DiffContrasts,
    HelmertContrasts,
    PolyContrasts,
    SumContrasts,
    TreatmentContrasts,
)
from .scale import scale


@stateful_transform
def standardize(
    x: Any,
    center: bool = True,
    rescale: bool = True,
    ddof: int = 0,
    _state: Optional[Dict[str, Any]] = None,
) -> Any:
    return scale(x, center=center, scale=rescale, ddof=ddof, _state=_state)


def Treatment(reference: Any = UNSET) -> TreatmentContrasts:
    return TreatmentContrasts(base=reference)


@stateful_transform
def Q(variable: str, _context: Optional[Mapping[str, Any]] = None) -> Any:
    return _context.data[variable]  # type: ignore


PATSY_COMPAT_TRANSFORMS = {
    "standardize": standardize,
    "Q": Q,
    "Treatment": Treatment,
    "Poly": PolyContrasts,
    "Sum": SumContrasts,
    "Helmert": HelmertContrasts,
    "Diff": DiffContrasts,
}
