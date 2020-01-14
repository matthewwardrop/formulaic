from collections.abc import Mapping
import itertools


class LayeredMapping(Mapping):

    def __init__(self, *layers):
        self.layers = self.__filter_layers(layers)

    @staticmethod
    def __filter_layers(layers):
        return [
            layer
            for layer in layers
            if layer is not None
        ]

    def __getitem__(self, key):
        for layer in self.layers:
            if key in layer:
                return layer[key]
        raise KeyError(key)

    def __iter__(self):
        keys = set()
        for layer in self.layers:
            for key in layer:
                if key not in keys:
                    keys.add(key)
                    yield key

    def __len__(self):
        return len(set(itertools.chain(*[list(layer) for layer in self.layers])))

    def with_layers(self, *layers, prepend=True, inplace=False):
        layers = self.__filter_layers(layers)
        if not layers:
            return self

        new_layers = [*layers, *self.layers] if prepend else [*self.layers, *layers]

        if inplace:
            self.layers = new_layers
            return self

        return LayeredMapping(*new_layers)
