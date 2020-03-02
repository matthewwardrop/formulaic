from collections.abc import MutableMapping
import itertools


class LayeredMapping(MutableMapping):

    def __init__(self, *layers):
        self.mutations = {}
        self.layers = self.__filter_layers(layers)

    @staticmethod
    def __filter_layers(layers):
        return [
            layer
            for layer in layers
            if layer is not None
        ]

    def __getitem__(self, key):
        for layer in [self.mutations, *self.layers]:
            if key in layer:
                return layer[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.mutations[key] = value

    def __delitem__(self, key):
        if key in self.mutations:
            del self.mutations[key]
        else:
            raise KeyError(f"Key '{key}' not found in mutable layer.")

    def __iter__(self):
        keys = set()
        for layer in [self.mutations, *self.layers]:
            for key in layer:
                if key not in keys:
                    keys.add(key)
                    yield key

    def __len__(self):
        return len(set(itertools.chain(list(self.mutations), *[list(layer) for layer in self.layers])))

    def with_layers(self, *layers, prepend=True, inplace=False):
        layers = self.__filter_layers(layers)
        if not layers:
            return self

        if inplace:
            self.layers = [*layers, *self.layers] if prepend else [*self.layers, *layers]
            return self

        new_layers = [*layers, self] if prepend else [self, *layers]
        return LayeredMapping(*new_layers)
