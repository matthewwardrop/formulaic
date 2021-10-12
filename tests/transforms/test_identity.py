from formulaic.transforms import identity


def test_identity():
    o = object()
    assert identity(o) is o
