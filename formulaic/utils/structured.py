from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from .sentinels import MISSING

_ItemType = TypeVar("_ItemType")
_SelfType = TypeVar("_SelfType", bound="Structured")


class Structured(Generic[_ItemType]):
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
        root: Any = MISSING,
        *,
        _metadata: Optional[Dict[str, Any]] = None,
        **structure: Any,
    ):
        if any(key.startswith("_") for key in structure):
            raise ValueError(
                "Substructure keys cannot start with an underscore. "
                f"The invalid keys are: {set(key for key in structure if key.startswith('_'))}."
            )
        if root is not MISSING:
            structure["root"] = root
        self._metadata = _metadata

        self._structure = {
            key: self.__prepare_item(key, item) for key, item in structure.items()
        }

    def __prepare_item(self, key: str, item: Any) -> Any:
        if isinstance(item, Structured):
            if type(item) is self.__class__:
                # Will already have been prepared
                return item
            return item._map(
                lambda x: self._prepare_item(key, x), as_type=self.__class__
            )
        if isinstance(item, tuple):
            return tuple(self.__prepare_item(key, v) for v in item)
        return self._prepare_item(key, item)

    def _prepare_item(self, key: str, item: Any) -> _ItemType:
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
        func: Union[
            Callable[[_ItemType], Any],
            Callable[[_ItemType, Tuple[Union[str, int], ...]], Any],
        ],
        recurse: bool = True,
        as_type: Optional[Type[Structured]] = None,
        _context: Tuple[Union[str, int], ...] = (),
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

        def apply_func(obj: Any, context: Tuple[Union[str, int], ...]) -> Any:
            if recurse and isinstance(obj, Structured):
                return obj._map(func, recurse=True, as_type=as_type, _context=context)
            if isinstance(obj, tuple):
                return tuple(apply_func(o, context + (i,)) for i, o in enumerate(obj))
            try:
                return func(obj, context)  # type: ignore
            except TypeError:
                return func(obj)  # type: ignore

        return (as_type or Structured)(
            **{
                key: apply_func(obj, _context + (key,))
                for key, obj in self._structure.items()
            }
        )

    def _flatten(self) -> Generator[_ItemType, None, None]:
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

        def do_recursion(obj: Any) -> Any:
            if recurse and isinstance(obj, Structured):
                return obj._to_dict()
            if isinstance(obj, tuple):
                return tuple(do_recursion(o) for o in obj)
            return obj

        return {key: do_recursion(value) for key, value in self._structure.items()}

    def _simplify(
        self: _SelfType,
        *,
        recurse: bool = True,
        unwrap: bool = True,
        inplace: bool = False,
    ) -> Union[_ItemType, _SelfType]:
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

        structure = structured._structure.copy()
        structure_modified: bool = False

        if recurse:

            def simplify_obj(
                obj: Union[_ItemType, Tuple[_ItemType], Structured[_ItemType]],
            ) -> Tuple[Union[_ItemType, Tuple[_ItemType], Structured[_ItemType]], bool]:
                """
                Return the simplified object, and a flag indicating whether the
                object was modified.
                """
                if isinstance(obj, Structured):
                    simplified = obj._simplify(recurse=True)
                    return simplified, simplified is not obj
                if isinstance(obj, tuple):
                    simplified = tuple(simplify_obj(o) for o in obj)
                    return tuple(s[0] for s in simplified), any(
                        s[1] for s in simplified
                    )
                return obj, False

            for key, value in tuple(structure.items()):
                value, value_modified = simplify_obj(value)
                if value_modified:
                    structure[key] = value
                    structure_modified = True

        if not inplace and not structure_modified:
            # Avoid any further work if simplification has not occurred
            return structured
        if not inplace:
            self = copy.copy(self)
        self._structure = structure
        return self

    def _update(self, root: Any = MISSING, **structure: Any) -> Structured[_ItemType]:
        """
        Return a new `Structured` instance that is identical to this one but
        the root and/or keys replaced with the nominated values.

        Args:
            root: The (optional) replacement of the root node.
            structure: Any additional key/values to update in the structure.
        """
        if root is not MISSING:
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

    @classmethod
    def _merge(
        cls,
        *objects: Any,
        merger: Optional[Callable] = None,
        _context: Tuple[str, ...] = (),
    ) -> Union[_ItemType, Structured[_ItemType], Tuple]:
        """
        Merge arbitrarily many objects into a single `Structured` instance.

        If any of `objects` are `Structured` or `tuple` instances, then all
        `objects` will be treated as `Structured` instances (being upcast as
        necessary) and then merged recursively; otherwise the objects will be
        merged directly by `merger`.

        Note: An empty set of objects will result in an empty `Structured`
        instance being returned.

        Args:
            objects: A tuple of Structured instances (will be upcast to a
                trivial `Structured` instance as necessary).
            merger: A callable which takes as arguments two or more items which
                are to be merged. If not provided, a basic fallback is provided
                that knows how to merge lists, dictionaries and sets.
            _context: A string representing the context of the merge. Intended
                for internal use.
        """
        if merger is None:
            merger = cls.__merger_default

        # If objects are not specified, return an empty `Structured` instance.
        if not objects:
            return cls()

        # Check for sequential (tuple) structures, and if so merge them and
        # return them wrapped in a `Structured` instance.
        all_tuples = all(isinstance(obj, tuple) for obj in objects)
        any_tuples = any(isinstance(obj, tuple) for obj in objects)

        if any_tuples and not all_tuples:
            raise ValueError(
                f"Substructures for `.{'.'.join(_context)}` are not aligned and cannot be merged."
            )

        if all_tuples:
            merged = tuple(itertools.chain(*objects))
            if _context:
                # We are merging substructure of `Structured` instances (and don't need the class wrapper)
                return merged
            return cls(merged)

        # Check whether all objects are not Structured instances (or tuples,
        # already excluded by above). If so, just call `merger` on them
        # directly.
        if all(not isinstance(obj, Structured) for obj in objects):
            return merger(*objects)  # type: ignore

        # Otherwise,iterate over objects, upcasting to `Structured` as necessary
        # and recursively merge them by merging their structure dictionaries.
        values_to_merge = defaultdict(list)

        for obj in objects:
            if isinstance(obj, Structured):
                for key, value in obj._structure.items():
                    values_to_merge[key].append(value)
            else:
                values_to_merge["root"].append(obj)

        return cls(
            **{
                key: (
                    cls._merge(*values, merger=merger, _context=_context + (key,))
                    if len(values) > 1
                    else values[0]
                )
                for key, values in values_to_merge.items()
            }
        )

    @staticmethod
    def __merger_default(*items: Any) -> Union[list, set, dict]:
        if all(isinstance(item, list) for item in items):
            return list(itertools.chain(*items))
        if all(isinstance(item, set) for item in items):
            return set.union(*items)
        if all(isinstance(item, dict) for item in items):
            return dict(itertools.chain(*(d.items() for d in items)))
        raise NotImplementedError(
            "The fallback `merger` for `Structured._merge` does not know how to "
            f"merge objects of types {repr(tuple(type(item) for item in items))}. "
            "Please specify `merger` explicitly."
        )

    def __dir__(self) -> List[str]:
        return [*super().__dir__(), *self._structure]

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr in self._structure:
            return self._structure[attr]
        raise AttributeError(
            f"This `{self.__class__.__name__}` instance does not have structure @ `{repr(attr)}`."
        )

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_"):
            super().__setattr__(attr, value)
            return
        self._structure[attr] = self.__prepare_item(attr, value)

    def __lookup_path(self, path: Tuple[Union[str, int], ...]) -> Any:
        obj = self
        idx = 0

        while idx < len(path):
            if isinstance(obj, Structured) and path[idx] in obj._structure:
                obj = obj._structure[cast(str, path[idx])]
            elif isinstance(obj, tuple) and isinstance(path[idx], int):
                obj = obj[path[idx]]
            else:
                break
            idx += 1
        else:
            return obj

        raise KeyError(
            f"Lookup {path} at index {idx} extends beyond structure of `{self.__class__.__name__}`."
        )

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            return self.__lookup_path(key)
        if self._has_root and not self._has_keys:
            return self.root[key]
        if key in (None, "root") and self._has_root:
            return self.root
        if isinstance(key, str) and not key.startswith("_") and key in self._structure:
            return self._structure[key]
        raise KeyError(
            f"This `{self.__class__.__name__}` instance does not have structure @ `{repr(key)}`."
        )

    def __setitem__(self, key: Any, value: Any) -> Any:
        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Cannot replace self.")
            obj = self.__lookup_path(key[:-1])
            if isinstance(obj, Structured):
                obj[key[-1]] = value
                return
            raise KeyError(
                f"Object @ {key[:-1]} is not a `Structured` instance. Unable to set value."
            )
        if not isinstance(key, str) or not key.isidentifier():
            raise KeyError(key)
        if key.startswith("_"):
            raise KeyError(
                "Substructure keys cannot start with an underscore. "
                f"The invalid keys are: {set(key for key in self._structure if key.startswith('_'))}."
            )
        self._structure[key] = self.__prepare_item(key, value)

    def __iter__(self) -> Generator[Any, None, None]:
        if self._has_root and not self._has_keys and isinstance(self.root, Iterable):  # pylint: disable=isinstance-second-argument-not-valid-type
            yield from self.root
        else:
            if self._has_root:  # Always yield root first.
                yield self.root
            for key, value in self._structure.items():
                if key != "root":
                    yield value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Structured):
            return self._structure == other._structure
        return False

    def __contains__(self, key: Any) -> bool:
        return key in self._structure

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __str__(self) -> str:
        return self.__repr__(to_str=str)  # type: ignore

    def __repr__(self, to_str: Callable[..., str] = repr) -> str:
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
