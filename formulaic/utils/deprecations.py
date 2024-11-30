import functools
import warnings
from typing import Callable, Optional, Tuple


def deprecated(
    func: Optional[Callable] = None,
    *,
    message: Optional[str] = None,
    as_of: Optional[Tuple[int, ...]] = None,
    removed_in: Optional[Tuple[int, ...]] = None,
) -> Callable:
    if func is None:
        return functools.partial(
            deprecated, message=message, as_of=as_of, removed_in=removed_in
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        warning = []
        if as_of is not None:
            warning.append(f"As of version {'.'.join(map(str, as_of))},")
        warning.append(message or f"the `{func.__name__}` method is deprecated.")
        if removed_in is not None:
            warning.append(
                f"This method will be removed in version {'.'.join(map(str, removed_in))}"
            )
        warnings.warn(" ".join(warning), DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper
