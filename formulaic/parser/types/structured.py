from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Optional,
    Iterable,
    TypeVar,
    Union,
)


ItemType = TypeVar("ItemType")
_MISSING = object()


class Structured(Generic[ItemType]):
    """
    Represents an arbitrarily nested structure.

    In the context of Formulaic, `Structured` is used to represent the structure
    of parsed formulae (e.g. separating the `lhs` and `rhs` of a formula), and
    the derived the model specs/matrices. Using `Structured` allows Formulaic to
    abstract out concerns regarding structured data, and focus on getting the
    containers right for singular use-cases.

    `Structured` instances have an (optional) root node (accessible via `.root`
    and `[None]`), and arbitrarily many named structured children. A
    `Structured` instance can be constructed using:
    ```
    Structured(<root object (optional)>, key=<child object>, key2=<another object>)
    ```
    The structure is then explored using <instance>.<key>.

    `Structured` instances also have a notion of "mapped attributes", which when
    accessed will recursively request the attribute on all children and return
    a new `Structured` instance with the result (and the same structure).

    Notes:
        - tuples have special meaning in `Structured` instances, and are treated
            part of the structure, and so when maps are applied, they tuples are
            iterated over (vs. lists which would be acted upon as a whole).
    """

    def __init__(
        self,
        root: Any = _MISSING,
        *,
        _mapped_attrs: Iterable[str] = None,
        _metadata: Dict[str, Any] = None,
        **structure,
    ):
        if any(key.startswith("_") for key in structure):
            raise ValueError(
                "Substructure keys cannot start with an underscore. "
                f"The invalid keys are: {set(key for key in structure if key.startswith('_'))}."
            )
        if root is not _MISSING:
            self.root = root
        self._structure = structure
        self._mapped_attrs = set(_mapped_attrs or ())
        self._metadata = _metadata

    @property
    def _has_root(self) -> bool:
        "Whether this instance of `Structured` has a root node."
        return hasattr(self, "root")

    def _map(
        self, func: Callable[[ItemType], Any], recurse: bool = True
    ) -> Structured[Any]:
        """
        Map a callable object onto all the structured objects, returning a
        `Structured` instance with identical structure, where the original
        objects are replaced with the output of `func`.

        Args:
            func: The callable to apply to all objects contained in the
                `Structured` instance.
            recurse: Whether to recursively map, or only map one level deep (the
                objects directly referenced by this `StructuredInstance`).
                When `True`, if objects within this structure are `Structured`
                instances also, then the map will be applied only on the leaf
                nodes (otherwise `func` will received `Structured` instances).
                (default: True).

        Returns:
            A `Structured` instance with the same structure as this instance,
            but with all objects transformed under `func`.
        """

        def apply_func(obj):
            if recurse and isinstance(obj, Structured):
                return obj._map(func, recurse=True)
            if isinstance(obj, tuple):
                return tuple(func(o) for o in obj)
            return func(obj)

        return Structured[ItemType](
            apply_func(self.root) if self._has_root else _MISSING,
            **{key: apply_func(obj) for key, obj in self._structure.items()},
        )

    def _to_dict(self, recurse: bool = True) -> Dict[Optional[str], Any]:
        """
        Generate a dictionary representation of this structure.

        The root node, if present, will have a `None` key assigned to it.

        Args:
            recurse: Whether to recursively convert any nested `Structured`
                instances into dictionaries also. If `False`, any nested
                `Structured` instances will be surfaced in the generated
                dictionary.

        Returns:
            The dictionary representation of this `Structured` instance.
        """

        def do_recursion(obj):
            if recurse and isinstance(obj, Structured):
                return obj._to_dict()
            return obj

        items = {}
        if self._has_root:
            items[None] = do_recursion(self.root)
        items.update(
            {key: do_recursion(value) for key, value in self._structure.items()}
        )
        return items

    def __dir__(self):
        return super().__dir__() + list(self._structure)

    def __getattr__(self, attr):
        if attr in ("__getstate__", "__setstate__"):
            raise AttributeError(attr)
        if not attr.startswith("_") and attr in self._structure:
            return self._structure[attr]
        if attr in self._mapped_attrs:
            return self._map(lambda x: getattr(x, attr))
        raise AttributeError(attr)

    def __getitem__(self, key):
        if key is None and self._has_root:
            return self.root
        if isinstance(key, str) and not key.startswith("_") and key in self._structure:
            return self._structure[key]
        if isinstance(key, int) and self._has_root and isinstance(self.root, tuple):
            return self.root[key]
        raise KeyError(key)

    def __iter__(self) -> Generator[Union[ItemType, Structured[ItemType]]]:
        if self._has_root:
            yield self.root
        yield from self._structure.values()

    def __len__(self) -> int:
        return int(self._has_root) + len(self._structure)

    def __str__(self):
        return self.__repr__(to_str=str)

    def __repr__(self, to_str=repr):
        import textwrap

        out = []
        for key, value in self._to_dict(recurse=False).items():
            if not key:
                out.append("root:")
            else:
                out.append(f".{key}")
            if isinstance(value, tuple):
                for i, obj in enumerate(value):
                    out.append(f"    [{i}]:")
                    out.append(textwrap.indent(to_str(obj), "        "))
            else:
                out.append(textwrap.indent(to_str(value), "    "))
            if not key and len(self) > 1:
                out.append("\nStructured children:\n")
        return "\n".join(out)
