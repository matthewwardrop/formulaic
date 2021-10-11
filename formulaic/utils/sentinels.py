class _MissingType:
    __instance__ = None

    def __new__(cls):
        if cls.__instance__ is None:
            cls.__instance__ = super(_MissingType, cls).__new__(cls)
        return cls.__instance__

    def __bool__(self):
        return False

    def __repr__(self):
        return "MISSING"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


MISSING = _MissingType()
