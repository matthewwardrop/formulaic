import itertools
from collections.abc import MutableMapping
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class LayeredMapping(MutableMapping):
    """
    A mutable mapping implementation that allows you to stack multiple mappings
    on top of one another, passing key lookups through the stack from top to
    bottom until the key is found or the stack is exhausted. Mutations are
    stored in an additional layer local only to the `LayeredMapping` instance,
    and the layers passed in are never mutated.
    """

    def __init__(self, *layers: Tuple[Optional[Mapping]]):
        """
        Crepare a `LayeredMapping` instance, populating it with the nominated
        layers.
        """
        self.mutations: Dict = {}
        self.layers: List[Mapping] = self.__filter_layers(layers)

    @staticmethod
    def __filter_layers(layers: Iterable[Mapping]) -> List[Mapping]:
        """
        Filter incoming `layers` down to those which are not null.
        """
        return [layer for layer in layers if layer is not None]

    def __getitem__(self, key: Any) -> Any:
        for layer in [self.mutations, *self.layers]:
            if key in layer:
                return layer[key]
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any):
        self.mutations[key] = value

    def __delitem__(self, key: Any):
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
        return len(
            set(itertools.chain(self.mutations, *[layer for layer in self.layers]))
        )

    def with_layers(
        self,
        *layers: Tuple[Optional[Mapping]],
        prepend: bool = True,
        inplace: bool = False,
    ) -> "LayeredMapping":
        """
        Return a copy of this `LayeredMapping` instance with additional layers
        added.

        Args:
            layers: The layers to add.
            prepend: Whether to add the layers before (if `True`) or after (if
                `False`) the current layers.
            inplace: Whether to mutate the existing `LayeredMapping` instance
                instead of returning a copy.

        Returns:
            A reference to the `LayeredMapping` instance with the extra layers.
        """
        layers = self.__filter_layers(layers)
        if not layers:
            return self

        if inplace:
            self.layers = (
                [*layers, *self.layers] if prepend else [*self.layers, *layers]
            )
            return self

        new_layers = [*layers, self] if prepend else [self, *layers]
        return LayeredMapping(*new_layers)
