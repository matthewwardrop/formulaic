import wrapt


class ModelMatrix(wrapt.ObjectProxy):
    def __init__(self, data, spec=None):
        wrapt.ObjectProxy.__init__(self, data)
        self._self_model_spec = spec

    @property
    def model_spec(self):
        return self._self_model_spec

    def __repr__(self):
        return self.__wrapped__.__repr__()  # pragma: no cover
