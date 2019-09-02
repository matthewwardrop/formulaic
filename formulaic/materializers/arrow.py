from interface_meta import override

from .pandas import PandasMaterializer


class ArrowMaterializer(PandasMaterializer):

    REGISTRY_NAME = 'arrow'
    DEFAULT_FOR = ['pyarrow.lib.Table']

    @override
    def _init(self, sparse=False):
        super()._init(sparse=False)
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

    def __contains__(self, value):
        return value in self.column_names

    def __getitem__(self, key):
        if key not in self.column_names:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = self.table.column(key).to_pandas()
        return self._cache[key]
