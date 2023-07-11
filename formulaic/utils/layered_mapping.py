from __future__ import annotations

import itertools
from collections.abc import MutableMapping
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

# Cached property was introduced in Python 3.8 (we currently support 3.7)
try:
    from functools import cached_property
except ImportError:  # pragma: no cover
    from cached_property import cached_property  # type: ignore


class LayeredMapping(MutableMapping):
    """
    A mutable mapping implementation that allows you to stack multiple mappings
    on top of one another, passing key lookups through the stack from top to
    bottom until the key is found or the stack is exhausted. Mutations are
    stored in an additional layer local only to the `LayeredMapping` instance,
    and the layers passed in are never mutated.

    Nested named layers can be extracted via attribute lookups, or via
    `.named_layers`.
    """

    def __init__(self, *layers: Optional[Mapping], name: Optional[str] = None):
        """
        Crepare a `LayeredMapping` instance, populating it with the nominated
        layers.
        """
        self.name = name
        self._mutations: Dict = {}
        self._layers: List[Mapping] = self.__filter_layers(layers)

    @staticmethod
    def __filter_layers(layers: Iterable[Optional[Mapping]]) -> List[Mapping]:
        """
        Filter incoming `layers` down to those which are not null.
        """
        return [layer for layer in layers if layer is not None]

    def __getitem__(self, key: Any) -> Any:
        for layer in [self._mutations, *self._layers]:
            if key in layer:
                return layer[key]
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self._mutations[key] = value

    def __delitem__(self, key: Any) -> None:
        if key in self._mutations:
            del self._mutations[key]
        else:
            raise KeyError(f"Key '{key}' not found in mutable layer.")

    def __iter__(self) -> Iterator[Any]:
        keys = set()
        for layer in [self._mutations, *self._layers]:
            for key in layer:
                if key not in keys:
                    keys.add(key)
                    yield key

    def __len__(self) -> int:
        return len(set(itertools.chain(self._mutations, *self._layers)))

    def with_layers(
        self,
        *layers: Optional[Mapping],
        prepend: bool = True,
        inplace: bool = False,
        name: Optional[str] = None,
    ) -> LayeredMapping:
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
            self._layers = (
                [*layers, *self._layers] if prepend else [*self._layers, *layers]
            )
            self.name = name
            if "named_layers" in self.__dict__:
                del self.named_layers
            return self

        new_layers = [*layers, self] if prepend else [self, *layers]
        return LayeredMapping(*new_layers, name=name)

    # Named layer lookups and caching

    @cached_property
    def named_layers(self) -> Dict[str, LayeredMapping]:
        """
        A mapping from string names to named `LayeredMapping` instances. If no
        children mappings are named, this will be an empty dictionary. If more
        than one layer shares the same name, only the first will be included.
        """
        named_layers = {}
        local = {}
        for layer in reversed(self._layers):
            if isinstance(layer, LayeredMapping):
                if layer.name:
                    local[layer.name] = layer
                named_layers.update(layer.named_layers)
        named_layers.update(local)
        if self.name:
            named_layers[self.name] = self
        return named_layers

    def get_with_layer_name(
        self, key: Any, default: Any = None, *, _path: Tuple[str, ...] = ()
    ) -> Tuple[Any, Optional[str]]:
        """
        Return the value for the nominated `key` (or `default` if `key` is not
        in this mapping); and the name of the layer from which the value is
        sourced. If the layer is unnamed, the name of the closest parent is
        used, or `None`.

        Args:
            key: The name of the key for which a value should be extracted.
            default: The default value to use if `key` is not found in this
                mapping.
            _path: The current path through layers when resolving things
                recursively. This is typically only used internally.
        """
        name = ":".join([*_path, self.name]) if self.name else (":".join(_path) or None)
        if key in self._mutations:
            return self._mutations[key], name
        for layer in self._layers:
            if key in layer:
                if isinstance(layer, LayeredMapping):
                    return layer.get_with_layer_name(
                        key, _path=(*_path, self.name) if self.name else _path
                    )
                return layer[key], name
        return default, None

    def get_layer_name_for_key(self, key: str) -> Optional[str]:
        """
        Return the name of the layer from which `key` would be extracted.

        Args:
            key: The name of the key for which the name of the layer hosting the
                value should be extracted.
        """
        return self.get_with_layer_name(key)[1]

    def __getattr__(self, attr: str) -> LayeredMapping:
        if attr not in self.named_layers:
            raise AttributeError(f"{repr(attr)} does not correspond to a named layer.")
        return self.named_layers[attr]
