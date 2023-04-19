from formulaic.utils.stateful_transforms import stateful_transform

from .contrasts import (
    TreatmentContrasts,
    PolyContrasts,
    SumContrasts,
    HelmertContrasts,
    DiffContrasts,
)
from .scale import scale


@stateful_transform
def standardize(x, center=True, rescale=True, ddof=0, _state=None):
    return scale(x, center=center, scale=rescale, ddof=ddof, _state=_state)


def Treatment(reference=TreatmentContrasts.MISSING):
    return TreatmentContrasts(base=reference)


PATSY_COMPAT_TRANSFORMS = {
    "standardize": standardize,
    "Treatment": Treatment,
    "Poly": PolyContrasts,
    "Sum": SumContrasts,
    "Helmert": HelmertContrasts,
    "Diff": DiffContrasts,
}
