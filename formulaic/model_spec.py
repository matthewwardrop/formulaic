import inspect
from .materializers import FormulaMaterializer


class ModelSpec:

    def __init__(self, formula, ensure_full_rank=True, feature_names=None, materializer=None, transforms=None, encoding=None):
        self.formula = formula
        self.ensure_full_rank = ensure_full_rank
        self.feature_names = feature_names
        self.materializer = materializer
        self.transforms = transforms or {}
        self.encoding = encoding or {}

    @property
    def materializer(self):
        return self._materializer

    @materializer.setter
    def materializer(self, materializer):
        if isinstance(materializer, FormulaMaterializer) or inspect.isclass(materializer) and issubclass(materializer, FormulaMaterializer):
            materializer = materializer.REGISTRY_NAME
        assert materializer is None or isinstance(materializer, str), materializer
        self._materializer = materializer

    def get_model_matrix(self, data, **kwargs):
        if self.materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(self.materializer)
        return materializer(data, **kwargs).get_model_matrix(self)
