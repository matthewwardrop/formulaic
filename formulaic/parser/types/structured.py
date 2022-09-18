from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)


ItemType = TypeVar("ItemType")
_MISSING = object()


class Structured(Generic[ItemType]):
    """
    Layers structure onto an arbitrary type.

    Structure can be added in two ways: by keys and by tuples, and can be
    arbitrarily nested. If present, the object assigned to the "root" key is
    treated specially, in that enumeration over the structured instance is
    equivalent to enumeration over the root node if there is no other structure.
    Otherwise, enumeration and key look up is done over the top-level values in
    the container in the order in which they were assigned (except that the root
    node is always first).

    The structure is mutable (new keys can be added, or existing attributes
    overridden) by direct assignment in the usual way; or via the `_update`
    method. To avoid collision with potential keys, all methods and attributes
    are preceded with an underscore. Contrary to Python convention, these are
    still considered public methods.

    Attributes:
        _structure: A dictionary of the keys stored in the `Structured`
            instance.
        _metadata: A dictionary of metadata which can be used to store arbitrary
            information about the `Structured` instance.

    Examples:
        ```
        >>>  s = Structured((1, 2), b=3, c=(4,5)); s
        root:
            [0]:
                1
            [1]:
                2
        .b:
            3
        .c:
            [0]:
                4
            [1]:
                5
        >>> list(s)
        [(1, 2), 3, (4, 5)]
        >>> s.root
        (1, 2)
        >>> s.b
        3
        >>> s._map(lambda x: x+1)
        root:
            [0]:
                2
            [1]:
                3
        .b:
            4
        .c:
            [0]:
                5
            [1]:
                6
        ```
    """

    __slots__ = ("_structure", "_metadata")

    def __init__(
        self,
        root: Any = _MISSING,
        *,
        _metadata: Dict[str, Any] = None,
        **structure,
    ):
        if any(key.startswith("_") for key in structure):
            raise ValueError(
                "Substructure keys cannot start with an underscore. "
                f"The invalid keys are: {set(key for key in structure if key.startswith('_'))}."
            )
        if root is not _MISSING:
            structure["root"] = self.__prepare_item("root", root)
        self._metadata = _metadata

        self._structure = {
            key: self.__prepare_item(key, item) for key, item in structure.items()
        }

    def __prepare_item(self, key: str, item: Any) -> ItemType:
        if isinstance(item, Structured):
            return item._map(
                lambda x: self._prepare_item(key, x), as_type=self.__class__
            )
        if isinstance(item, tuple):
            return tuple(self.__prepare_item(key, v) for v in item)
        return self._prepare_item(key, item)

    def _prepare_item(self, key: str, item: Any) -> ItemType:
        return item

    @property
    def _has_root(self) -> bool:
        """
        Whether this instance of `Structured` has a root node.
        """
        return "root" in self._structure

    @property
    def _has_keys(self) -> bool:
        """
        Whether this instance of `Structured` has any non-root named
        substructures.
        """
        return set(self._structure) != {"root"}

    @property
    def _has_structure(self) -> bool:
        """
        Whether this instance of `Structured` has any non-trivial structure,
        including named or unnamed substructures.
        """
        return self._has_keys or self._has_root and isinstance(self.root, tuple)

    def _map(
        self,
        func: Callable[[ItemType], Any],
        recurse: bool = True,
        as_type: Optional[Type[Structured]] = None,
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
            as_type: An optional subclass of `Structured` to use for the mapped
                values. If not provided, the base `Structured` type is used.

        Returns:
            A `Structured` instance with the same structure as this instance,
            but with all objects transformed under `func`.
        """

        def apply_func(obj):
            if recurse and isinstance(obj, Structured):
                return obj._map(func, recurse=True, as_type=as_type)
            if isinstance(obj, tuple):
                return tuple(apply_func(o) for o in obj)
            return func(obj)

        return (as_type or Structured)(
            **{key: apply_func(obj) for key, obj in self._structure.items()}
        )

    def _flatten(self) -> Generator[ItemType]:
        """
        Flatten any nested structure into a sequence of all values stored in
        this `Structured` instance. The order is currently that yielded by a
        depth-first iteration, however this is not guaranteed and should not
        be relied upon.
        """
        for value in self._structure.values():
            if isinstance(value, Structured):
                yield from value._flatten()
            elif isinstance(value, tuple):
                for v in value:
                    if isinstance(v, Structured):
                        yield from v._flatten()
                    else:
                        yield v
            else:
                yield value

    def _to_dict(self, recurse: bool = True) -> Dict[Optional[str], Any]:
        """
        Generate a dictionary representation of this structure.

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
            if isinstance(obj, tuple):
                return tuple(do_recursion(o) for o in obj)
            return obj

        return {key: do_recursion(value) for key, value in self._structure.items()}

    def _simplify(
        self, *, recurse: bool = True, unwrap: bool = True, inplace: bool = False
    ) -> Union[Any, Structured[ItemType]]:
        """
        Simplify this `Structured` instance by:
            - returning the object stored at the root node if there is no other
                structure (removing as many `Structured` wrappers as satisfy
                this requirement).
            - if `recurse` is `True`, recursively applying the logic above to
                any nested `Structured` instances.

        Args:
            unwrap: Whether to unwrap the root node (returning the raw
                unstructured root value) if there is no other structure.
            recurse: Whether to recurse the simplification into the objects
                associated with the keys of this (and nested) `Structured`
                instances.
            inplace: Whether to simplify the current structure (`True`), or
                return a new object with the simplifications (`False`). Note
                that if `True`, `unwrap` *must* be `False`.
        """
        if inplace and unwrap:
            raise RuntimeError(
                f"Cannot simplify `{self.__class__.__name__}` instances "
                "in-place if `unwrap` is `True`."
            )
        structured = self
        while (
            isinstance(structured, Structured)
            and structured._has_root
            and not structured._has_structure
            and (unwrap or isinstance(structured.root, Structured))
        ):
            structured = structured.root

        if not isinstance(structured, Structured):
            return structured

        structure = structured._structure

        if recurse:

            def simplify_obj(obj):
                if isinstance(obj, Structured):
                    return obj._simplify(recurse=True)
                if isinstance(obj, tuple):
                    return tuple(simplify_obj(o) for o in obj)
                return obj

            structure = {
                key: simplify_obj(value) for key, value in structured._structure.items()
            }

        if inplace:
            self._structure = structure
            return self
        return self.__class__(
            _metadata=self._metadata,
            **structure,
        )

    def _update(self, root=_MISSING, **structure) -> Structured[ItemType]:
        """
        Return a new `Structured` instance that is identical to this one but
        the root and/or keys replaced with the nominated values.

        Args:
            root: The (optional) replacement of the root node.
            structure: Any additional key/values to update in the structure.
        """
        if root is not _MISSING:
            structure["root"] = root
        return self.__class__(
            **{
                "_metadata": self._metadata,
                **self._structure,
                **{
                    key: self.__prepare_item(key, item)
                    for key, item in structure.items()
                },
            }
        )

    def __dir__(self):
        return super().__dir__() + list(self._structure)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr in self._structure:
            return self._structure[attr]
        raise AttributeError(
            f"This `{self.__class__.__name__}` instance does not have structure @ `{repr(attr)}`."
        )

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            super().__setattr__(attr, value)
            return
        self._structure[attr] = self.__prepare_item(attr, value)

    def __getitem__(self, key):
        if self._has_root and not self._has_keys:
            return self.root[key]
        if key in (None, "root") and self._has_root:
            return self.root
        if isinstance(key, str) and not key.startswith("_") and key in self._structure:
            return self._structure[key]
        raise KeyError(
            f"This `{self.__class__.__name__}` instance does not have structure @ `{repr(key)}`."
        )

    def __setitem__(self, key, value):
        if not isinstance(key, str) or not key.isidentifier():
            raise KeyError(key)
        if key.startswith("_"):
            raise KeyError(
                "Substructure keys cannot start with an underscore. "
                f"The invalid keys are: {set(key for key in self._structure if key.startswith('_'))}."
            )
        self._structure[key] = self.__prepare_item(key, value)

    def __iter__(self) -> Generator[Union[ItemType, Structured[ItemType]]]:
        if self._has_root and not self._has_keys and isinstance(self.root, Sequence):
            yield from self.root
        else:
            if self._has_root:  # Always yield root first.
                yield self.root
            for key, value in self._structure.items():
                if key != "root":
                    yield value

    def __eq__(self, other):
        if isinstance(other, Structured):
            return self._structure == other._structure
        return False

    def __contains__(self, key):
        return key in self._structure

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __str__(self):
        return self.__repr__(to_str=str)

    def __repr__(self, to_str=repr):
        import textwrap

        d = self._to_dict(recurse=False)
        keys = [key for key in d if key != "root"]
        if self._has_root:
            keys.insert(0, "root")

        out = []
        for key in keys:
            if key == "root":
                out.append("root:")
            else:
                out.append(f".{key}:")
            value = d[key]
            if isinstance(value, tuple):
                for i, obj in enumerate(value):
                    out.append(f"    [{i}]:")
                    out.append(textwrap.indent(to_str(obj), "        "))
            else:
                out.append(textwrap.indent(to_str(value), "    "))
        return "\n".join(out)
