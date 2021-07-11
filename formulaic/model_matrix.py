import wrapt


class ModelMatrix(wrapt.ObjectProxy):

    def __init__(self, data, spec=None, metadata=None):
        wrapt.ObjectProxy.__init__(self, data)
        self._self_model_spec = spec
        self._self_metadata = metadata or {}

    @property
    def model_spec(self):
        return self._self_model_spec

    @property
    def metadata(self):
        return self._self_metadata

    def __repr__(self):
        return self.__wrapped__.__repr__()  # pragma: no cover
