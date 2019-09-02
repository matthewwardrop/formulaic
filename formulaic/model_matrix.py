import wrapt


class ModelMatrix(wrapt.ObjectProxy):

    __slots__ = ('formula', 'materializer')

    def __init__(self, formula, data, feature_names=None, materializer=None):
        wrapt.ObjectProxy.__init__(self, data)
        self.formula = formula
        self.feature_names = feature_names
        self.materializer = materializer
