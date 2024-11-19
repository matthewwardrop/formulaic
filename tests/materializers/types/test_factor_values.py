import copy
import pickle

from formulaic.materializers.types import FactorValues


def test_factor_values_copy():
    d = {"1": object()}
    f = FactorValues(d, drop_field="test")

    f2 = copy.copy(f)
    assert f2.__wrapped__ == d
    assert f2.__wrapped__["1"] is d["1"]
    assert f2.__formulaic_metadata__.drop_field == "test"

    f3 = copy.deepcopy(f)
    assert f3.__wrapped__["1"] is not d["1"]
    assert f3.__formulaic_metadata__.drop_field == "test"


def test_factor_values_pickle():
    d = {"1": object()}
    f = FactorValues(d, drop_field="test")

    f2 = pickle.loads(pickle.dumps(f))

    assert list(f2.__wrapped__.keys()) == list(d.keys())
    assert type(f2.__wrapped__["1"]) is type(d["1"])
    assert f2.__formulaic_metadata__.drop_field == "test"
