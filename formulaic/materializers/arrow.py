from interface_meta import override

import pandas

from .pandas import PandasMaterializer


class ArrowMaterializer(PandasMaterializer):

    REGISTER_NAME = "arrow"
    REGISTER_INPUTS = ("pyarrow.lib.Table",)

    @override
    def _init(self):
        self.__data_context = LazyArrowTableProxy(self.data)

    @override
    @property
    def data_context(self):
        return self.__data_context


class LazyArrowTableProxy:
    def __init__(self, table):
        self.table = table
        self.column_names = set(self.table.column_names)
        self._cache = {}
        self.index = pandas.RangeIndex(len(table))

    def __contains__(self, value):
        return value in self.column_names

    def __getitem__(self, key):
        if key not in self.column_names:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = self.table.column(key).to_pandas()
        return self._cache[key]
