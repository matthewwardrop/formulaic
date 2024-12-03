import copy

from formulaic.utils.sentinels import MISSING, UNSET, Sentinel


def test_missing():
    assert MISSING is Sentinel.MISSING
    assert UNSET is Sentinel.UNSET
    assert MISSING is not UNSET
    assert bool(MISSING) is False
    assert repr(MISSING) == "MISSING"
    assert copy.copy(MISSING) is MISSING
    assert copy.deepcopy(MISSING) is MISSING
