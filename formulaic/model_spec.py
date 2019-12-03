from collections import OrderedDict
import inspect

from .formula import Formula
from .materializers import FormulaMaterializer


class ModelSpec:

    def __init__(self, formula, ensure_full_rank=True, structure=None, materializer=None, transforms=None, encoding=None):
        self.formula = Formula.from_spec(formula)
        self.ensure_full_rank = ensure_full_rank
        self.structure = structure
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

    @property
    def feature_names(self):
        return [
            name
            for row in self.structure
            for name in row[2]
        ]

    @property
    def feature_indices(self):
        return OrderedDict([
            (name, i)
            for i, name in enumerate(self.feature_names)
        ])

    @property
    def term_slices(self):
        slices = OrderedDict()
        start = 0
        for row in self.structure:
            end = start + len(row[2])
            slices[row[0]] = slice(start, end)
        return slices

    def get_model_matrix(self, data, **kwargs):
        if self.materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(self.materializer)
        return materializer(data, **kwargs).get_model_matrix(self)
