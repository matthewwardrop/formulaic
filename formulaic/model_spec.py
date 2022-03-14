from collections import OrderedDict
import inspect

from .formula import Formula
from .materializers import FormulaMaterializer, NAAction


class ModelSpec:
    def __init__(
        self,
        formula,
        ensure_full_rank=True,
        structure=None,
        materializer=None,
        na_action="drop",
        output=None,
        transform_state=None,
        encoder_state=None,
    ):
        self.formula = Formula.from_spec(formula)
        self.ensure_full_rank = ensure_full_rank
        self.structure = structure
        self.materializer = materializer
        self.na_action = NAAction(na_action)
        self.output = output
        self.transform_state = transform_state if transform_state is not None else {}
        self.encoder_state = encoder_state if encoder_state is not None else {}

    @property
    def materializer(self):
        return self._materializer

    @materializer.setter
    def materializer(self, materializer):
        if (
            isinstance(materializer, FormulaMaterializer)
            or inspect.isclass(materializer)
            and issubclass(materializer, FormulaMaterializer)
        ):
            materializer = materializer.REGISTER_NAME
        assert materializer is None or isinstance(materializer, str), materializer
        self._materializer = materializer

    @property
    def feature_names(self):
        return [name for row in self.structure for name in row[2]]

    @property
    def feature_indices(self):
        return OrderedDict([(name, i) for i, name in enumerate(self.feature_names)])

    @property
    def term_slices(self):
        slices = OrderedDict()
        start = 0
        for row in self.structure:
            end = start + len(row[2])
            slices[row[0]] = slice(start, end)
            start = end
        return slices

    def get_model_matrix(self, data, **kwargs):
        if self.materializer is None:
            materializer = FormulaMaterializer.for_data(data)
        else:
            materializer = FormulaMaterializer.for_materializer(self.materializer)
        return materializer(data, **kwargs).get_model_matrix(self)

    def differentiate(self, *vars, use_sympy=False):
        return ModelSpec(
            formula=self.formula.differentiate(*vars, use_sympy=use_sympy),
            ensure_full_rank=self.ensure_full_rank,
            structure=self.structure,
            materializer=self.materializer,
            transform_state=self.transform_state,
            encoder_state=self.encoder_state,
        )
