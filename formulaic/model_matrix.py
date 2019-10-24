import wrapt


class ModelMatrix(wrapt.ObjectProxy):

    __slots__ = ('formula', 'materializer')

    def __init__(self, data, spec=None):
        wrapt.ObjectProxy.__init__(self, data)
        self.model_spec = spec
