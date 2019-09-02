from collections.abc import Mapping
import itertools


class LayeredContext(Mapping):

    def __init__(self, *layers):
        self.layers = layers or []

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
        return len(set(itertools.chain(list(layer) for layer in self.layers)))
